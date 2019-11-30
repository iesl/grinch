/* Copyright (C) 2019 University of Massachusetts Amherst.
   This file is part of “grinch”
   http://github.com/iesl/grinch
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

package grinch.core

import java.io.{File, PrintWriter}
import java.util.concurrent.ExecutorService
import java.util.PriorityQueue

import cc.factorie.util.JavaHashSet

import scala.collection.mutable.ArrayBuffer
import grinch.nn.NSW
import grinch.utils.ComparisonCounter

import scala.collection.mutable

/**
  * Grinch clustering class, maintains cluster tree,
  * performs tree re-arrangements.
  * @param conf - Configuration object
  */
class Grinch(conf: Config) {

  val config = conf

  val can_forget_points = false

  var leaf_counter = 0

  var root: GrinchNode = null

  var cached_knn_search: IndexedSeq[(Float, GrinchNode)] = new ArrayBuffer[(Float, GrinchNode)](k())
  var collapsibles: PriorityQueue[(Float, GrinchNode)] = if (this.config.max_num_leaves == -1) null else new PriorityQueue[(Float,GrinchNode)](GPQComparator)

  var deleted_leaves = JavaHashSet[GrinchNode]()

  val _nn_struct = new NSW(config.k,config.nsw_r,exact_nn = config.exact_nn) //new Exact()

  // statistics tracked
  var number_of_swaps = 0
  var number_of_restructs = 0
  var number_of_grafts = 0
  var number_of_rotates = 0
  var this_number_of_swaps = 0
  var this_number_of_restructs = 0
  var this_number_of_grafts = 0
  var this_number_of_rotates = 0
  var time_in_search = 0.0
  var time_in_rotate = 0.0
  var time_in_graft = 0.0
  var time_in_restruct = 0.0
  var time_in_coll = 0.0
  var time_overall = 0
  var this_time_in_search = 0.0
  var this_time_in_rotate = 0.0
  var this_time_in_graft = 0.0
  var this_time_in_restruct = 0.0
  var this_time_overall = 0
  var this_max_height = 0
  var number_of_points_seen = 0
  var number_of_attempted_grafts = 0
  var this_pt_id = "-1"
  var total_number_of_grafts_considered = 0
  val printMaxHeight = false
  val printGraftHeights = false
  val acceptedHeights = new ArrayBuffer[Int]()
  val rejectedHeights = new ArrayBuffer[Int]()
  var graftsAcceptedAfterSinglElim = 0
  var graftsConsideredAfterSinglElim = 0
  val printGraftsAfterSE = false
  val acceptedAfterSE = new ArrayBuffer[Int]()
  val consideredAfterSE = new ArrayBuffer[Int]()
  val acceptedOverall = new ArrayBuffer[Int]()
  val consideredOverall = new ArrayBuffer[Int]()
  val wordPrints = false
  var wordPrintsSoFar = 0
  var write_graft_ids = false
  var graft_ids = new ArrayBuffer[String]

  var save_all_trees = false
  var save_all_trees_dir = "/tmp/all_trees/"
  if (save_all_trees) {
    new File(save_all_trees_dir).mkdirs()
  }

  /**
    * Construct a grinch tree on the points
    * @param points
    */
  def build_dendrogram(points: Iterator[Point]) = {
    var sum_times = 0.0
    var last_time = 0.0
    val start = System.currentTimeMillis()
    points.zipWithIndex.foreach {
      case (pt, idx) =>
        if (idx % 100 == 0) {
          println(f"Pts=${this.number_of_points_seen}\tLvs=${this.leaf_counter}\tTot=${sum_times}%.2f\tAvg=${ if (this.number_of_points_seen > 0) sum_times / this.number_of_points_seen else 0.0}%.5f\tLast=${last_time}%.3f" +
            f"\tSearch_Time=${this.time_in_search}%.2f\tRotate_Time=${this.time_in_rotate}%.2f\tGraft_Time=${this.time_in_graft}%.2f\tRestruct_Time=${this.time_in_restruct}%.2f\tColl_Time=${this.time_in_coll}%.2f" +
            f"\tRot/Graft/Swap\t${this.number_of_rotates}\t${this.number_of_grafts}\t${this.number_of_swaps}\t${ComparisonCounter.count}\tMaxHeight=${this.this_max_height}")
        }
        val start = System.currentTimeMillis()
        this.insert(pt)
        val end = System.currentTimeMillis()
        last_time = (end-start).toFloat / 1000.0
        sum_times += (end-start).toFloat / 1000.0
        this.clear_this_stats()
    }
    println(s"Pts=${this.number_of_points_seen}\tAvg=${ if (this.number_of_points_seen > 0) sum_times / this.number_of_points_seen else 0.0}\tLast=${last_time}\tRot/Graft/Swap\t${this.number_of_rotates}\t${this.number_of_grafts}\t${this.number_of_swaps}\t${ComparisonCounter.count}\tMaxHeight=${this.this_max_height}")
    val end = System.currentTimeMillis()
    println(s"Total Time: ${end-start}")
    println(s"Number of grafts accepted ${number_of_grafts}")
    println(s"Total Grafts Considered ${total_number_of_grafts_considered}")
    println(s"Number of Grafts Accepted after SE ${graftsAcceptedAfterSinglElim}")
    println(s"Number of Grafts Considered after SE ${graftsConsideredAfterSinglElim}")
    if (printGraftHeights)
      write_graft_heights()
    if (printGraftsAfterSE)
      write_grafts_after_SE()
  }

  /**
    * Add one point to the Grinch Tree
    * @param pt
    */
  def insert(pt: Point) = {

    if (printGraftsAfterSE) {
      consideredAfterSE += 0
      acceptedAfterSE += 0
      consideredOverall += 0
      acceptedOverall += 0
    }

    this.this_pt_id = pt.pid
    this.number_of_points_seen += 1
    val new_node = this.node_from_pt(pt)
    if (this.config.graft_debug) {
      println(s"${new java.util.Date().toString} [insert] new_node=${new_node.id} pt=(${pt.pid},${pt.value})")
    }
    val search_start = System.currentTimeMillis()
    val nn = this.cknn_and_insert(new_node, this.k(), JavaHashSet[GrinchNode]())
    val search_end = System.currentTimeMillis()
    this.time_in_search += (search_end - search_start).toFloat / 1000.0
    this.this_time_in_search = (search_end - search_start).toFloat / 1000.0
    this.cached_knn_search = nn
    if (nn == null || this.root == null) {
      assert(this.root == null)
      this.root = new_node
    } else {

      if (this.config.graft_debug) {
        println(s"${new java.util.Date().toString} [insert] nn=${nn.head._2.id} nn.score=${nn.head._1}")
      }
      val rotate_start = System.currentTimeMillis()
      assert(nn.head._2.isRoot || !nn.head._2.parent.isCollapsed)
      val sib = this.rotate(new_node, nn.head._2)
      if (this.config.graft_debug) {
        println(s"${new java.util.Date().toString} [insert] Nearest neighbors ${nn.map(f => (f._1, f._2.pt.pid)).mkString(", ")}")
        println(s"${new java.util.Date().toString} #nearestNeighbor\t${pt.pid}\t${nn.head._2.pt.pid}")
      }
      val rotate_end = System.currentTimeMillis()
      this.time_in_rotate += (rotate_end - rotate_start).toFloat / 1000.0
      this.this_time_in_rotate = (rotate_end - rotate_start).toFloat / 1000.0
      val parent = this.node_from_nodes(sib, new_node)
      if (this.config.graft_debug) {
        println(s"${new java.util.Date().toString} #new_parent_node\t${parent.id}")
      }
      sib.make_sibling(new_node, parent)
      this.root = this.root.root()
      var curr_update = parent
      while (curr_update != null) {
        curr_update.updatedFromChildren()
        curr_update = curr_update.parent
      }
      if (this.collapsibles != null) {
        if (parent != null && parent.validCollapse()) {
          this.collapsibles.add((parent.score(), parent))
        }
      }

      // Print trees in the rotate case:
      if (save_all_trees) {
        this.root.serializeTree(new File(save_all_trees_dir,s"pt_${this.number_of_points_seen}_rotate.tsv"))
      }

      val graft_start = System.currentTimeMillis()
      if (this.config.perform_graft) {
        this.graft(parent)
      }
      val graft_end = System.currentTimeMillis()

      if (save_all_trees && this_number_of_grafts > 0) {
        this.root.serializeTree(new File(save_all_trees_dir,s"pt_${this.number_of_points_seen}_graft.tsv"))
      }

      this.time_in_graft += (graft_end - graft_start).toFloat / 1000.0
      this.this_time_in_graft = (graft_end - graft_start).toFloat / 1000.0
      this.root = this.root.root()
      this.leaf_counter += 1
      if (this.collapsibles != null) {
        if (this.leaf_counter > this.config.max_num_leaves) {
          //          println(s"${new java.util.Date().toString} [insert] collapsing! number of leaves ${this.leaf_counter} > max num leaves ${this.config.max_num_leaves}")

          var res = this.collapsibles.poll()
          var score = res._1
          var to_collapse = res._2
          while (!to_collapse.validCollapse()) {
            val res = this.collapsibles.poll()
            score = res._1
            to_collapse = res._2
          }

          //          println(s"${new java.util.Date().toString} [insert] collapsing! node ${to_collapse.id} with ${to_collapse.leaves().size} collapsed leaves")
          val collapsedLeaves = to_collapse.collapse()
          collapsedLeaves.foreach {
            l =>
              this.deleted_leaves.add(l)
          }
          this.leaf_counter -= 1

          if (to_collapse.parent.validCollapse()) {
            this.collapsibles.add(to_collapse.parent.score(),to_collapse.parent)
          }
        }
      }
      if (printMaxHeight) {
        this.this_max_height = this.root.compute_max_height()
      }

    }
  }

  def node_from_nodes(n1: GrinchNode, n2: GrinchNode) = {
    val ent = n1.ent.copy()
    ent.mergedRep(n2.ent)
    val n = new GrinchNode(ent.d, config.exact_dist_thresh, ent, config, this)
    ent.gnode = n
    n.pointCounter = n1.pointCounter + n2.pointCounter
    n
  }

  def node_from_pt(pt: Point) = {
    val ent = new Ent(pt.value.length)
    ent.observe(pt)
    val n = new GrinchNode(ent.d, config.exact_dist_thresh, ent, config, this)
    n.addPt(pt)
    n.ent.gnode = n
    n
  }

  /* * *                                                                 * * *
   * * *                   Nearest Neighbor Methods                      * * *
   * * *                                                                 * * */

  def nn_struct() = _nn_struct

  def k() = this.config.k

  def cknn_and_insert(ent: GrinchNode, k: Int, offlimits: mutable.Set[GrinchNode]): IndexedSeq[(Float, GrinchNode)] = nn_struct().cknn_and_insert(ent, k, offlimits)

  def cknn(ent: GrinchNode, k: Int, offlimits: mutable.Set[GrinchNode]): IndexedSeq[(Float, GrinchNode)] = nn_struct().cknn(ent, k, offlimits)

  def graft_cknn_search(curr: GrinchNode, k: Int, offlimits: mutable.Set[GrinchNode]) = {
    if (!this.config.single_graft_search) {
      this.cknn(curr, k, offlimits)
    } else {
      if (config.graft_debug) {
        println(s"${new java.util.Date().toString} [graft_cnn_search] using cached search! ")
      }
      val res = new ArrayBuffer[(Float, GrinchNode)]()
      this.cached_knn_search.foreach {
        case (score, node) =>
          if (!offlimits.contains(node)) {
            res.+=((score, node))
          }
      }
      this.cached_knn_search = res
      res
    }
  }

  def update_collapsibles(new_node: GrinchNode) = {
    if (collapsibles ne null) {
      if (new_node.validCollapse()) {
        val score = new_node.score()
        collapsibles.add((score, new_node))
      }
    }
  }

  /* * *                                                                 * * *
   * * *                   Tree Rearrangements                           * * *
   * * *                                                                 * * */


  /**
    * This defines the linkage function
    * @param one
    * @param nn
    * @return
    */
  def e_score(one: Ent, nn: Ent): Float = throw new NotImplementedError()

  def graft(gnode: GrinchNode) = {
    if (this.config.graft_debug) {
      println(s"${new java.util.Date().toString} [graft] grafting from ${gnode.id}")
    }
    var curr = gnode
    val start_offlimits = System.currentTimeMillis()
    val offlimits = JavaHashSet[GrinchNode]()
    curr.siblings().foreach(s => offlimits.add(s))
    curr.leaves().foreach(l => offlimits.add(l))
    offlimits.add(curr)
    val end_offlimits = System.currentTimeMillis()

    if (this.config.graft_debug) {
      println(s"${new java.util.Date().toString} [graft] Offlimits size ${offlimits.size}")
      println(s"${new java.util.Date().toString} [graft] Offlimits Time ${end_offlimits - start_offlimits}")
    }

    var graft_attempts = 0
    var exit_on_beam = false
    var exit_on_single_elimination = false
    var exit_on_graft_cap = false
    var exit_on_lca = false
    var exit_on_no_nn = false
    var number_of_grafts_considered = 0
    var thinkWeShouldExit = false

    var mutualRejectThisTime = false


    while (curr != null && curr.parent != null
      && !exit_on_beam && curr.pointCounter < this.config.graft_size_cap
      && !exit_on_single_elimination && !exit_on_graft_cap
      && !exit_on_lca && !exit_on_no_nn) {

      assert(!thinkWeShouldExit)
      if (this.config.graft_debug) {
        println(s"${new java.util.Date().toString} Grafting curr = ${curr.id} curr.parent ${curr.parent.id}")
      }
      var performed_graft = false
      var prev_curr = curr
      val nn_search_start = System.currentTimeMillis()
      val nns = this.graft_cknn_search(curr, this.k(), offlimits)
      val nn_search_end = System.currentTimeMillis()

      if (nns.nonEmpty) {
        if (this.config.graft_debug) {
          var (nn_score, nn) = nns.head
          println(s"${new java.util.Date().toString} [graft] Nearest neighbors ${nns.map(s => s"(${s._1},${s._2.id})").mkString(", ")}")
          println(s"${new java.util.Date().toString} [graft] Number of Nearest neighbors ${nns.size}")
          println(s"${new java.util.Date().toString} [graft] Found Nearest Neighbor ${nn.id} with score ${nn_score} in ${nn_search_end - nn_search_start}")
        }

        var (nn_score, nn) = nns.head
        val start_lca = System.currentTimeMillis()
        val lca = curr.lca(nn)
        val end_lca = System.currentTimeMillis()
        if (this.config.graft_debug) {
          println(s"${new java.util.Date().toString} [graft] lca ${lca.id} curr ${curr.id} nn ${nn.id}")
        }

        var isDone = false
        while (curr != lca && nn != lca
          && !nn.siblings().contains(curr)
          && !isDone) {
          assert(!thinkWeShouldExit)
          if (this.config.graft_debug) {
            println(s"${new java.util.Date().toString} [graft] curr.point_counter ${curr.pointCounter} and nn.point_counter ${nn.pointCounter}")
            println(s"${new java.util.Date().toString} [graft] curr.parent.point_counter ${curr.parent.pointCounter} and nn.parent.point_counter ${nn.parent.pointCounter}")
          }
          if (curr.pointCounter > this.config.graft_size_cap ||
            nn.pointCounter > this.config.graft_size_cap) {
            exit_on_graft_cap = true
            isDone = true
          } else {
            val start_compute_score = System.currentTimeMillis()
            val score_if_grafted = this.e_score(curr.ent, nn.ent)
            val curr_parent_score = curr.parent.score()
            val nn_parent_score = nn.parent.score()
            val end_compute_score = System.currentTimeMillis()
            if (this.config.graft_debug) {
              println(s"${new java.util.Date().toString} [graft] time to compute scores: ${end_compute_score - start_compute_score}")
            }
            val i_like_you = score_if_grafted > curr_parent_score
            val you_like_me = score_if_grafted > nn_parent_score
            if (this.config.graft_debug) {
              println(s"${new java.util.Date().toString} [graft] i_like_you=$i_like_you\tyou_like_me=$you_like_me\tscore_if_grafted=$score_if_grafted\tcurr_parent_score=$curr_parent_score\tnn_parent_score=$nn_parent_score")
            }
            if (this.config.single_elimination && !i_like_you && !you_like_me) {
              exit_on_single_elimination = true
              if (this.config.graft_debug) {
                println(s"${new java.util.Date().toString} [debug] exist on single elimination=${exit_on_single_elimination}")
              }
              isDone = true
            } else {
              if (!you_like_me) {

                if (mutualRejectThisTime) {
                  graftsConsideredAfterSinglElim += 1
                  if (printGraftsAfterSE) {
                    consideredAfterSE(consideredAfterSE.size - 1) += 1
                    consideredOverall(consideredOverall.size - 1) += 1
                  }
                }

                if (!i_like_you) {
                  mutualRejectThisTime = true
                }

                if (printGraftHeights)
                  rejectedHeights += curr.pointCounter
                nn = nn.parent
                if (this.config.graft_debug) {
                  println(s"${new java.util.Date().toString} [graft] you_like_me=${you_like_me}\tnn_is_now=$nn")
                }
                number_of_grafts_considered += 1
                total_number_of_grafts_considered +=1

              } else if (you_like_me && !i_like_you) {

                if (mutualRejectThisTime) {
                  graftsConsideredAfterSinglElim += 1
                  if (printGraftsAfterSE) {
                    consideredAfterSE(consideredAfterSE.size - 1) += 1
                    consideredOverall(consideredOverall.size - 1) += 1
                  }
                } else {
                  if (printGraftsAfterSE) {
                    consideredOverall(consideredOverall.size - 1) += 1
                  }
                }

                if (printGraftHeights)
                  rejectedHeights += curr.pointCounter
                curr = curr.parent
                if (this.config.graft_debug) {
                  println(s"${new java.util.Date().toString} [graft] you_like_me=${you_like_me}\ti_like_you=${i_like_you}\tcurr_is_now=$curr")
                }
                number_of_grafts_considered += 1
                total_number_of_grafts_considered  += 1

              } else {
                assert(you_like_me)
                assert(i_like_you)

                if (total_number_of_grafts_considered != config.skip_graft_at) {

                  if (write_graft_ids)
                    graft_ids += s"${total_number_of_grafts_considered}"

                  if (wordPrints) {

                    println(s"WORD PRINT ${wordPrintsSoFar}")
                    println(s"GRAFTING:\n  curr ${curr.descLeafLabels.mkString("\t")}\n  and nn: ${nn.descLeafLabels.mkString("\t")} \n" +
                      s"from ${curr.sibling().descLeafLabels.mkString("\t")} \nfrom: ${nn.sibling().descLeafLabels.mkString("\t")}")

                    this.root.root().serializeTree(new File(s"/tmp/words/tree_before_${wordPrintsSoFar}.gv"))

                  }

                  if (this.config.graft_debug) {
                    println(s"${new java.util.Date().toString} [graft] you_like_me=${you_like_me}\ti_like_you=${i_like_you}\tcurr_is_now=$curr")
                  }
                  val nn_grandparent = nn.parent.parent

                  performed_graft = true

                  if (printGraftHeights)
                    acceptedHeights += curr.pointCounter

                  number_of_grafts += 1
                  this_number_of_grafts += 1

                  if (mutualRejectThisTime) {
                    graftsAcceptedAfterSinglElim += 1
                    graftsConsideredAfterSinglElim += 1
                    if (printGraftsAfterSE) {
                      acceptedAfterSE(acceptedAfterSE.size - 1) += 1
                      consideredAfterSE(consideredAfterSE.size - 1) += 1
                      acceptedOverall(acceptedOverall.size - 1) += 1
                      consideredOverall(consideredOverall.size - 1) += 1
                    }
                  } else {
                    if (printGraftsAfterSE) {
                      acceptedOverall(acceptedOverall.size - 1) += 1
                      consideredOverall(consideredOverall.size - 1) += 1
                    }
                  }

                  val z = curr.sibling()
                  val l_in_zees_children = z.children.contains(nn)

                  if (this.config.graft_debug) {
                    println(s"${new java.util.Date().toString} [graft] nn_grandparent=${if (nn_grandparent != null) nn_grandparent.id else "None"}\tz=${z.id}")
                  }

                  val start_node_construction = System.currentTimeMillis()
                  val parent = this.node_from_nodes(curr, nn)
                  curr.make_sibling(nn, parent)
                  val end_node_construction = System.currentTimeMillis()
                  if (this.config.graft_debug) {
                    println(s"${new java.util.Date().toString} [graft] node construction time ${end_node_construction - start_node_construction}")
                  }
                  // Update

                  val start_update = System.currentTimeMillis()
                  var curr_update: GrinchNode = null
                  List(nn_grandparent, curr.parent).foreach {
                    case start =>
                      curr_update = start
                      while (curr_update != null) {
                        curr_update.updatedFromChildren()
                        curr_update = curr_update.parent
                      }
                  }
                  val end_update = System.currentTimeMillis()
                  if (this.config.graft_debug) {
                    println(s"${new java.util.Date().toString} [graft] update time ${end_update - start_update}")
                  }

                  val start_coll_time = System.currentTimeMillis()
                  if (this.collapsibles != null) {
                    if (parent != null && parent.validCollapse()) {
                      this.collapsibles.add((parent.score(), parent))
                    }
                    if (nn_grandparent != null && nn_grandparent.validCollapse()) {
                      this.collapsibles.add((nn_grandparent.score(),nn_grandparent))
                    }
                  }

                  val end_coll_time = System.currentTimeMillis()
                  this.time_in_coll += (end_coll_time - start_coll_time).toFloat / 1000.0
                  // Restruct
                  val restruct_start = System.currentTimeMillis()
                  if (this.config.perform_restruct) {
                    if (!l_in_zees_children) {
                      this.restructure(z, curr.lca(z))
                      this.number_of_restructs += 1
                      this.this_number_of_restructs += 1
                    } else {
                      if (this.config.graft_debug) {
                        println(s"${new java.util.Date().toString} [restruct] l_in_zees_children=True")
                      }
                    }
                  }
                  val restruct_end = System.currentTimeMillis()
                  if (this.config.graft_debug) {
                    println(s"${new java.util.Date().toString} [graft] restructure time ${restruct_end - restruct_start} seconds")
                  }
                  this.time_in_restruct += (restruct_end - restruct_start).toFloat / 1000.0
                  this.this_time_in_restruct += (restruct_end - restruct_start).toFloat / 1000.0

                  if (lca == this.root) {
                    this.root = curr.root()
                    if (this.config.graft_debug) {
                      println(s"${new java.util.Date().toString} [graft] setting new root ${this.root}")
                    }
                  }

                  if (wordPrints) {
                    print(s"IDS: ${curr.id}\t${nn.id}")
                    print(s"New point: ${gnode.children.map(_.id)}")
                    println(s"WORD PRINT ${wordPrintsSoFar}")
                    println(s"GRAFTING:\n  curr ${curr.descLeafLabels.mkString("\t")}\n  and nn: ${nn.descLeafLabels.mkString("\t")} \n" +
                      s"from ${curr.sibling().descLeafLabels.mkString("\t")} \nfrom: ${nn.sibling().descLeafLabels.mkString("\t")}")

                    this.root.root().serializeTree(new File(s"/tmp/words/tree_after_${wordPrintsSoFar}.gv"))
                    wordPrintsSoFar += 1
                  }

                  isDone = true
                }
              }
            }
          }
        }
        if (performed_graft) {
          curr = curr.parent
          if (this.config.graft_debug) {
            println(s"${new java.util.Date().toString} [graft] PERFORMED GRAFT curr=${curr.id}")
          }
        } else {
          curr = lca
          exit_on_lca = true
          if (this.config.graft_debug) {
            println(s"${new java.util.Date().toString} [graft] NO GRAFT (LCA). curr=${curr.id}")
          }
        }
      } else {
        curr = curr.parent
        if (this.config.single_graft_search && this.cached_knn_search.nonEmpty) {
          exit_on_no_nn = true
        }
        if (this.config.graft_debug) {
          println(s"${new java.util.Date().toString} [graft] NO GRAFT (NO NN) curr=${curr.id}")
        }
      }

      if (graft_attempts < this.config.graft_beam) {
        exit_on_beam = true
        if (this.config.graft_debug) {
          println(s"${new java.util.Date().toString} [graft] exist on beam ")
        }
      }

      if (curr != null && curr.parent != null
        && curr.pointCounter < config.graft_size_cap
        && !exit_on_beam && !exit_on_single_elimination
        && !exit_on_graft_cap && !exit_on_lca && !exit_on_no_nn) {
        val start_offlimits = System.currentTimeMillis()
        curr.parent.leaves_excluding(Set(prev_curr)).foreach {
          case l =>
            offlimits.add(l)
        }
        val end_offlimits = System.currentTimeMillis()
        if (this.config.graft_debug) {
          println(s"${new java.util.Date().toString} [graft] len(offlimits)=${offlimits.size} in ${end_offlimits - start_offlimits} s")
        }
      } else {
        if (this.config.graft_debug) {
          println(s"${new java.util.Date().toString} [graft] skipping offlimits because we are going to exit graft ")
          thinkWeShouldExit = true
        }
      }
    }
    if (this.config.graft_debug) {
      if (graft_attempts >= this.config.graft_beam) {
        println(s"${new java.util.Date().toString} [graft] Returning... graft_attempts_beam_met=${graft_attempts}")
      }
      if (exit_on_single_elimination) {
        println(s"${new java.util.Date().toString} [graft] Returning... exit_on_single_elimination=${exit_on_single_elimination}")
      }
      if (exit_on_graft_cap) {
        println(s"${new java.util.Date().toString} [graft] Returning... exit_on_graft_cap=${exit_on_graft_cap}")
      }
      println(s"${new java.util.Date().toString} [graft] Number of grafts considered ${number_of_grafts_considered}")

    }
  }

  def restructure(z: GrinchNode, r: GrinchNode) = {
    if (config.graft_debug) {
      println(s"${new java.util.Date().toString} [restruct] z=${z.id} r=${r.id}")
    }
    var curr = z
    val start_find_ancestors = System.currentTimeMillis()
    val r_ancs = r.ancestors().toSet[GrinchNode]
    val end_find_ancestors = System.currentTimeMillis()
    if (config.graft_debug) {
      println(s"${new java.util.Date().toString} [restruct] time to find r ancesntors ${end_find_ancestors - start_find_ancestors}")
    }
    while (curr != r) {
      val start_curr_anc = System.currentTimeMillis()
      val ancs = curr.ancestors_with_point_limit(this.config.restruct_size_cap)
      val end_curr_anc = System.currentTimeMillis()
      if (config.graft_debug) {
        println(s"${new java.util.Date().toString} [restruct] time to find the curr ancestors ${end_curr_anc - start_curr_anc} ]")
      }
      val start_curr_anc_sib = System.currentTimeMillis()
      val ancs_sibs = ancs.flatMap(a => if (a.parent != null && a.pointCounter < this.config.restruct_size_cap) Some(a.sibling()) else None)
      val end_curr_anc_sib = System.currentTimeMillis()

      if (config.graft_debug) {
        println(s"${new java.util.Date().toString} [restruct] time to find the curr ancestors siblings ${end_curr_anc_sib - start_curr_anc_sib} ]")
      }
      val start_anc_scores = System.currentTimeMillis()
      val anc_sibs_scores = ancs_sibs.map(s => (this.e_score(z.ent, s.ent), s))
      val end_anc_scores = System.currentTimeMillis()
      if (config.graft_debug) {
        println(s"${new java.util.Date().toString} [restruct] time to score the ancestors sibs ${end_anc_scores - start_anc_scores} ]")
      }

      val start_find_max = System.currentTimeMillis()
      var best_score = Float.NegativeInfinity
      var best_n = null.asInstanceOf[GrinchNode]
      anc_sibs_scores.foreach {
        case (score, n) =>
          if (score > best_score) {
            best_score = score
            best_n = n
          }
      }
      val end_find_max = System.currentTimeMillis()
      if (config.graft_debug) {
        println(s"${new java.util.Date().toString} [restruct] time to find the max score ${end_find_max - start_find_max} ]")
        println(s"${new java.util.Date().toString} [restruct] best_score=${best_score}, curr.score()=${curr.parent.score()}")
      }
      if (best_score > curr.parent.score()) {
        if (config.graft_debug) {
          println(s"${new java.util.Date().toString} [restruct] perform swap ${curr.sibling().id}, ${best_n.id}")
        }
        curr.sibling().swap(best_n)
        this.number_of_swaps += 1
        this.this_number_of_swaps += 1
      }
      curr = curr.parent
    }
  }

  def rotate(gnode: GrinchNode, sib: GrinchNode) = {
    if (this.config.graft_debug) {
      println(s"${new java.util.Date().toString} [_find_insert] gnode=${gnode.id}\tsib=${sib.id}")
    }
    var curr = sib
    if (this.config.perform_rotate) {
      var score = this.e_score(gnode.ent, curr.ent)
      var curr_parent_score = if (curr.parent != null) curr.parent.score() else Float.NegativeInfinity
      while (curr.isDeleted || (curr.parent != null && score < curr_parent_score && curr.parent.pointCounter <= this.config.rotation_size_cap)) {
        curr = curr.parent
        curr_parent_score = if (curr.parent != null) curr.parent.score() else Float.NegativeInfinity
        score = this.e_score(gnode.ent, curr.ent)
        this.number_of_rotates += 1
        this.this_number_of_rotates += 1
      }
      if (this.config.graft_debug) {
        println(s"${new java.util.Date().toString} [_find_insert] RETURN score=${score}\tcurr=${curr.id}\tcurr.parent.score=${curr_parent_score}")
        if (curr.parent != null && curr.parent.pointCounter >= this.config.rotation_size_cap) {
          println(s"${new java.util.Date().toString} [_find_insert] rotation_size_cap=${this.config.rotation_size_cap}\tcurr.pointCounter=${curr.pointCounter}\tcurr.parent.pointCounter=${curr.parent.pointCounter}")
        }
      }
    }
    curr
  }

  /* * *                                                                 * * *
   * * *                   Tracking Stats                                * * *
   * * *                                                                 * * */


  def clear_this_stats() = {
    this_number_of_swaps = 0
    this_number_of_restructs = 0
    this_number_of_grafts = 0
    this_number_of_rotates = 0
    this_time_in_search = 0
    this_time_in_rotate = 0
    this_time_in_graft = 0
    this_time_in_restruct = 0
    this_time_overall = 0
    this_pt_id = "-1"
  }


  def write_grafts_after_SE() = {
    val pw = new PrintWriter("/tmp/accepted_se.txt")
    acceptedAfterSE.foreach {
      h =>
        pw.println(h)
    }
    pw.close()
    val pw2 = new PrintWriter("/tmp/considered_se.txt")
    consideredAfterSE.foreach {
      h =>
        pw2.println(h)
    }
    pw2.close()


    val pw3 = new PrintWriter("/tmp/accepted_ovr.txt")
    acceptedOverall.foreach {
      h =>
        pw3.println(h)
    }
    pw3.close()
    val pw4 = new PrintWriter("/tmp/considered_ovr.txt")
    consideredOverall.foreach {
      h =>
        pw4.println(h)
    }
    pw4.close()

  }

  def write_graft_heights() = {
    val pw = new PrintWriter("/tmp/accepted.txt")
    acceptedHeights.foreach {
      h =>
        pw.println(h)
    }
    pw.close()
    val pw2 = new PrintWriter("/tmp/rejected.txt")
    rejectedHeights.foreach {
      h =>
        pw2.println(h)
    }
    pw2.close()
  }


}



class CosLinkGrinch(config: Config) extends Grinch(config) {

  override val can_forget_points = true

  override def e_score(one: Ent, nn: Ent): Float = {
    if (one.needsUpdate) {
      one.update()
    }
    if (nn.needsUpdate) {
      nn.update()
    }
    cosSim(one.sum,nn.sum)
  }

  /**
    * Euclidean distance between x and y, ||x-y||
    *
    * @param x Vector 1
    * @param y Vector 2
    * @return Distance
    */
  def cosSim(x: Array[Float], y: Array[Float]): Float = {
    ComparisonCounter.increment()
    var i = 0
    val len = x.length
    assert(y.length == len)
    var res = 0.0f
    var x_norm = 0.0f
    var y_norm = 0.0f
    while (i < len) {
      res += x(i) * y(i)
      x_norm += x(i) * x(i)
      y_norm += y(i) * y(i)
      i += 1
    }
    if (x_norm > 0 && y_norm > 0) {
      res / (math.sqrt(x_norm) * math.sqrt((y_norm))).toFloat
    } else {
      0.0f
    }
  }
}


class AvgLinkGrinch(config: Config) extends Grinch(config) {
  override def e_score(one: Ent, nn: Ent): Float = {
    if (one.needsUpdate) {
      one.update()
    }
    if (nn.needsUpdate) {
      nn.update()
    }
    var d = 0.0f
    var z = 0.0f
    var i = 0
    while (i < one.points.length) {
      var j = 0
      val pt1 = one.points(i)
      while (j < nn.points.length) {
        val pt2 = nn.points(j)
        d += minusNorm(pt1.value, pt2.value)
        z += 1.0f
        j += 1
      }
      i += 1
    }
    d / z
  }

  /**
    * Euclidean distance between x and y, ||x-y||
    *
    * @param x Vector 1
    * @param y Vector 2
    * @return Distance
    */
  def minusNorm(x: Array[Float], y: Array[Float]): Float = {
    ComparisonCounter.increment()
    var i = 0
    val len = x.length
    assert(y.length == len)
    var res = 0.0f
    while (i < len) {
      val r = x(i) - y(i)
      res += r * r
      i += 1
    }
    -res
  }
}


class ParAvgLinkGrinch(config: Config, nswThreadpool:ExecutorService) extends AvgLinkGrinch(config) {
  override def cknn(ent: GrinchNode, k: Int, offlimits: mutable.Set[GrinchNode]): IndexedSeq[(Float, GrinchNode)] =  nn_struct().cknn(ent, k, offlimits,nswThreadpool)

  override def cknn_and_insert(ent: GrinchNode, k: Int, offlimits: mutable.Set[GrinchNode]): IndexedSeq[(Float, GrinchNode)] = nn_struct().cknn_and_insert(ent, k, offlimits, nswThreadpool)
}

class ParCosLinkGrinch(config: Config, nswThreadpool:ExecutorService) extends CosLinkGrinch(config) {
  override def cknn(ent: GrinchNode, k: Int, offlimits: mutable.Set[GrinchNode]): IndexedSeq[(Float, GrinchNode)] =  nn_struct().cknn(ent, k, offlimits,nswThreadpool)

  override def cknn_and_insert(ent: GrinchNode, k: Int, offlimits: mutable.Set[GrinchNode]): IndexedSeq[(Float, GrinchNode)] = nn_struct().cknn_and_insert(ent, k, offlimits, nswThreadpool)
}