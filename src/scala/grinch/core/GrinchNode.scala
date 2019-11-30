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
import java.util.{Comparator, PriorityQueue, UUID}

import cc.factorie.util.{JavaHashSet, Threading}

import scala.collection.mutable.ArrayBuffer
import grinch.utils.ConsistentId

import scala.collection.mutable

class GrinchNode(dim: Int, exactDistThreshold: Int = 20,
                 initEnt: Ent, config: Config, g: Grinch)
  extends Comparable[GrinchNode] {

  def eScoreFn(node1: GrinchNode, node2: GrinchNode): Float = g.e_score(node1.ent, node2.ent)

  /**
    * Unique id of PerchNode
    */
  val id: UUID = ConsistentId.nextId

  var parent: GrinchNode = null

  val grinch = g

  lazy val pt = pts.head

  var children = new ArrayBuffer[GrinchNode](2)
  var neighbors = new ArrayBuffer[GrinchNode](config.k)
  var edges_to_me = new ArrayBuffer[GrinchNode](config.k)

  val collapsedLeaves = new ArrayBuffer[GrinchNode]()

  var pts = new ArrayBuffer[Point](exactDistThreshold)

  var pointCounter = 0
  var isCollapsed = false

  def descLeafLabels = this.leaves().map(_.pt.label).toSet[String]

  /**
    * The siblings of this node
    *
    * @return Sequence of siblings
    */
  def siblings(): Seq[GrinchNode] = {
    if (this.parent != null)
      this.parent.children.filterNot(_ == this)
    else
      Seq()
  }

  /**
    * Write the tree in the evaluation file format:
    * Node Id \t Parent Id \t Label
    *
    * ParentId will be "None" if the node is the root
    * Label will be "None" for every internal node
    * and the ground truth class label for leaf nodes.
    *
    * @param file The file to write to
    */
  def serializeTree(file: File): Unit = {
    val pw = new PrintWriter(file, "UTF-8")
    val queue = new scala.collection.mutable.Queue[GrinchNode]()
    queue.enqueue(this)
    var currNode = this
    while (queue.nonEmpty) {
      currNode = queue.dequeue()
      val nodeId = if (currNode.isLeaf && currNode.collapsedLeaves.isEmpty) currNode.pts.head.pid else currNode.id.toString
      val parId = if (currNode.parent != null) currNode.parent.id else "None"
      val pid = if (currNode.isLeaf && currNode.collapsedLeaves.isEmpty) currNode.pts.head.label else "None"
      pw.println(s"$nodeId\t$parId\t$pid")
      currNode.children.foreach {
        c =>
          queue.enqueue(c)
      }
      currNode.collapsedLeaves.foreach {
        c =>
          queue.enqueue(c)
      }
    }
    pw.close()
  }

  def swap(other: GrinchNode) = {
    val self_parent = this.parent
    self_parent.children.-=(this)
    other.parent.children.-=(other)
    this.parent = other.parent
    other.parent = self_parent
    this.parent.children.+=(this)
    other.parent.children.+=(other)
  }

  def sibling(): GrinchNode = siblings().head

  /**
    * The aunts of this node
    *
    * @return Sequence of aunts
    */
  def aunts(): Seq[GrinchNode] = {
    if (this.parent != null && this.parent.parent != null) {
      this.parent.parent.children.filterNot(_ == this.parent)
    } else
      Seq()
  }

  /**
    * The root of this Perch cluster tree
    *
    * @return The root
    */
  def root(): GrinchNode = {
    var currNode = this
    while (!currNode.isRoot)
      currNode = currNode.parent
    currNode
  }

  def lca(other: GrinchNode): GrinchNode = {
    if (this.id == other.id) {
      return this
    }

    if (this.root() == this) {
      println(s"this.root() ${this.root().id} ${other.root().id}")
      assert(this.root() == other.root())
      return this
    }

    val ancs = JavaHashSet[UUID]()
    ancs.add(this.id)
    this.ancestors().foreach(a => ancs.add(a.id))

    var curr_node = other
    while (!ancs.contains(curr_node.id)) {
      curr_node = curr_node.parent
    }
    return curr_node
  }

  def compute_max_height() = {
    val lvs = this.root().leaves()
    lvs.map(_.ancestors().size).max
  }

  def collapse() = {
    // before collapsing, update one last time to make it real
    this.updatedFromChildren()
    this.ent.update() // set ent
    this.score() // cache score
    this.collapsedLeaves ++= this.leaves()
    this.collapsedLeaves.foreach(n => n.parent = this)
    this.isCollapsed = true
    assert(this.children.size == 2)
    assert(this.children(0).isLeaf)
    assert(this.children(1).isLeaf)

//    this.grinch._nn_struct.nodes.foreach {
//      n =>
//        assert(!n.neighbors.contains(this))
//    }
//    this.grinch._nn_struct.nodes.foreach {
//      n =>
//        if (n.neighbors.contains(this.children(0))) {
//          assert(this.children(0).edges_to_me.contains(n))
//        }
//        if (n.neighbors.contains(this.children(1))) {
//          assert(this.children(1).edges_to_me.contains(n))
//        }
//    }
    this.grinch._nn_struct.deleteNode(this.children(0))
    this.grinch._nn_struct.replaceNodeWith(this.children(1),this)
//    this.grinch._nn_struct.nodes.foreach {
//      n =>
//        assert(!n.neighbors.contains(this.children(0)))
//        assert(!n.neighbors.contains(this.children(1)))
//    }
    // add this new node to the nsw
//    this.grinch.cknn_and_insert(this,this.grinch.k(),mutable.HashSet[GrinchNode]())

    this.children.clear()
    assert(this.children.isEmpty)
    this.collapsedLeaves
  }

  def ancestors_with_point_limit(limit: Int) = {
    val anc = new ArrayBuffer[GrinchNode](100)
    // todo: better estimate of initial size
    var curr = this
    while (curr.parent != null && curr.parent.pointCounter < limit) {
      anc += curr.parent
      curr = curr.parent
    }
    anc
  }

  def ancestors(): Seq[GrinchNode] = {
    val anc = new ArrayBuffer[GrinchNode](100)
    // todo: better estimate of initial size
    var curr = this
    while (curr.parent != null) {
      anc += curr.parent
      curr = curr.parent
    }
    anc
  }

  def ancestorsIterator(): Iterator[GrinchNode] = {
    var curr = this

    new Iterator[GrinchNode] {

      override def hasNext: Boolean = curr != null

      override def next(): GrinchNode = {
        val res = curr.parent
        curr = curr.parent
        res
      }

    }
  }

  /**
    * Find all of the descendant leaves of this node.
    *
    * @return Sequence of leaves
    */
  def leaves(include_collap: Boolean = false): Seq[GrinchNode] = {
    val lvs = new ArrayBuffer[GrinchNode](this.pointCounter)
    val queue = new java.util.LinkedList[GrinchNode]()
    queue.push(this)
    while (!queue.isEmpty) {
      val n = queue.pop()
      if (n.collapsedLeaves.nonEmpty && include_collap) {
        assert(n.children.isEmpty)
        lvs ++= n.collapsedLeaves
      }
      else if (n.isLeaf)
        lvs += n
      else
        n.children.foreach {
          c =>
            queue.push(c)
        }
    }
    lvs
  }

  def leaves_excluding(excluding: Set[GrinchNode], incl_collap_lvs: Boolean = false, exclude_collapsed_nodes: Boolean = true) = {
    val lvs = new ArrayBuffer[GrinchNode](this.pointCounter)
    val queue = new java.util.LinkedList[GrinchNode]()
    queue.push(this)
    while (!queue.isEmpty) {
      val n = queue.pop()
      if (n.collapsedLeaves.nonEmpty && incl_collap_lvs) {
        assert(n.children.isEmpty)
        lvs ++= n.collapsedLeaves
      }
      else if (n.isLeaf) // && (!exclude_collapsed_nodes || !n.isCollapsed))
        lvs += n
      else
        n.children.foreach {
          c =>
            if (!excluding.contains(c)) {
              queue.push(c)
            }

        }
    }
    lvs
  }

  /**
    * Determine if this nodes is a leaf in the tree
    *
    * @return True/false if the node is a leaf
    */
  def isLeaf: Boolean = this.children.isEmpty

  /**
    * Determine if this node is the root of the tree
    *
    * @return True/false if the node is the root
    */
  def isRoot: Boolean = this.parent == null

  var isDeleted = false

  /**
    * Determine if this node is allowed to be collapsed.
    *
    * @return True/false if the node can be collapsed
    */
  def validCollapse(): Boolean = {
    !this.isDeleted && this.children.nonEmpty && this.children(0).isLeaf && this.children(1).isLeaf
  }


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   *                          Util Methods                             *
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /**
    * Break any ordering ties by using the id numbers of the nodes.
    *
    * @param o Another perch node
    * @return Comparison by id
    */
  override def compareTo(o: GrinchNode): Int = this.id.compareTo(o.id)


  def score() = {
    if (this._score.isEmpty) {
      this._score = Some(this.eScoreFn(this.children(0), this.children(1)))
    }
    this._score.get
  }

  var ent: Ent = initEnt
  var _score: Option[Float] = None

  def updatedFromChildren() = {
    assert(!this.isCollapsed)
    this.pointCounter = this.children(0).pointCounter + this.children(1).pointCounter
    this._score = None
    this.ent.gnode = this
    this.ent.needsUpdate = true
    this._score = None
  }

  def make_sibling(sibling: GrinchNode, parent: GrinchNode) = {
    assert(!parent.isCollapsed)
    val sib_parent = sibling.parent
    if (sibling.parent != null) {
      val sib_gp = sibling.parent.parent
      val old_sib = sibling.sibling()
      old_sib.parent = sib_gp
      if (sib_gp != null) {
        sib_gp.children -= sib_parent
        sib_gp.children += old_sib
      }
      sib_parent.parent = null
      sib_parent.children = new ArrayBuffer[GrinchNode](2)
      sib_parent.isDeleted = true
      sib_parent.pts = null
    } else {
      assert(sibling.children.isEmpty)
    }
    parent.parent = this.parent
    if (parent.parent != null) {
      parent.parent.children -= this
      parent.parent.children += parent
    }
    parent.children += this
    parent.children += sibling
    this.parent = parent
    sibling.parent = parent
  }

  def addPt(pt: Point): Unit = {
    this.pointCounter += 1
    this.pts.append(pt)
  }


  var accepting_neighbors = true
  def add_link(other: GrinchNode) = {
    if (this.accepting_neighbors) {
      this.neighbors.append(other)
      other.edges_to_me.append(this)
      if (config.nsw_debug) {
        println(s"${new java.util.Date().toString}\tself.add_neighbor(other}\tself=${this.id}\tother=${other.id}\tlen(self.neighbors)=${this.neighbors.size}")
      }
      if (this.neighbors.size == this.config.max_degree) {
        this.accepting_neighbors = false
      }
    }

    if (other.accepting_neighbors) {
      other.neighbors.append(this)
      this.edges_to_me.append(other)
      if (config.nsw_debug) {
        println(s"${new java.util.Date().toString}\tself.add_neighbor(other}\tself=${this.id}\tother=${other.id}\tlen(other.neighbors)=${other.neighbors.size}")
      }
      if (other.neighbors.size == this.config.max_degree) {
        other.accepting_neighbors = false
      }
    }

  }

  def score_neighbors(query: GrinchNode, offlimits: mutable.Set[GrinchNode], path: mutable.Set[GrinchNode], threadpool : ExecutorService = null) = {

    if (threadpool == null) {
      this.neighbors.filter( n=> !offlimits.contains(n) && !path.contains(n)).map(n => (this.eScoreFn(query,n),n))
    } else {
      Threading.parMap(this.neighbors,threadpool)(n => {
        if (!offlimits.contains(n) && !path.contains(n))
          Some((this.eScoreFn(query,n),n))
        else
          None
      }).flatten
    }
  }

  def cknn(query: GrinchNode,k:Int =1, offlimits:mutable.Set[GrinchNode], path: mutable.Set[GrinchNode],threadpool : ExecutorService = null): PriorityQueue[(Float,GrinchNode)] = {

    val local_path = JavaHashSet[GrinchNode]()
    val knn_not_offlimits = new PriorityQueue[(Float,GrinchNode)](GPQComparator)
    var curr = this
    var score = this.eScoreFn(curr,query)
    var best_score = score
    var best_node = curr
    var numSearched = 0

    assert(!offlimits.contains(curr))

    knn_not_offlimits.add((score,curr))
    while (true) {
      if (path.contains(curr)) {
        if (this.config.graft_debug) {
          println(s"${new java.util.Date().toString} [cknn] Returning None because curr was on the path ${curr.id}")
          return new PriorityQueue[(Float, GrinchNode)](GPQComparator)
        }
      }
      local_path.add(curr)
      val score_neighbors = curr.score_neighbors(query, offlimits, path,threadpool)
      score_neighbors.foreach {
        case (s,c) =>
          if (best_node == null || best_score < s || (best_score == s && best_node.id.compareTo(c.id) < 0)) {
            if (this.config.graft_debug) {
              println(s"${new java.util.Date().toString} [cknn] #NewBest\tscore=${s}\tc=${c}")
            }
            best_score = s
            best_node = c
          }
          if (knn_not_offlimits.size() == k) {
            val popped = knn_not_offlimits.poll()
//            println(s"Popped ${popped}")
//            println(knn_not_offlimits.iterator().asScala.mkString(" "))
          }
          knn_not_offlimits.add((s,c))
          numSearched += 1
      }
      while (knn_not_offlimits.size > k) {
        val popped = knn_not_offlimits.poll()
      }
      if (best_node == curr) {
        if (this.config.graft_debug) {
          println(s"${new java.util.Date().toString} [cknn] #SearchEnd\tscore=${best_score}\tc=${best_node}")
        }
        local_path.foreach( l => path.add(l))
        return knn_not_offlimits
      } else if (numSearched > config.maxSearchTime) {
        if (this.config.graft_debug) {
          println(s"${new java.util.Date().toString} [cknn] max search time met. returning #SearchEnd\tscore=${best_score}\tc=${best_node}")
        }
        local_path.foreach( l => path.add(l))
        return knn_not_offlimits
      } else {
        curr = best_node
        if (this.config.graft_debug) {
          println(s"${new java.util.Date().toString} [cknn] #SearchCont\tscore=${best_score}\tc=${best_node}")
        }
      }
    }
    assert(false, "should not be here")
    return null
  }


}

/**
  * Comparator used for various priorities in PerchNode class
  */
object GPQComparator extends Comparator[(Float, GrinchNode)] {
  override def compare(o1: (Float, GrinchNode), o2: (Float, GrinchNode)): Int = Ordering.Tuple2[Float, GrinchNode].compare(o1, o2)
}
