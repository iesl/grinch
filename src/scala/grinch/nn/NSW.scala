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

package grinch.nn

import java.util.concurrent.ExecutorService

import cc.factorie.util.{JavaHashMap, JavaHashSet, Threading}
import grinch.core.GrinchNode

import scala.collection.mutable
import scala.util.Random
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

class NSW(val k: Int, val r: Int, val seed: Int = 1451, exact_nn: Boolean= false) {


  var num_neighbor_edges = 0
  var nodes = new ArrayBuffer[GrinchNode]()
  var approx_max_degree = 0
  val random = new Random(seed)

  val exact = new Exact()

  def replaceNodeWith(v: GrinchNode, replacement: GrinchNode) = {
    v.edges_to_me.foreach{
      n =>
        n.neighbors -= v
        n.neighbors += replacement
        n.accepting_neighbors = (n.neighbors.size < n.grinch.config.max_degree)
    }
    v.neighbors.foreach{
      n =>
        n.edges_to_me -= v
        n.edges_to_me += replacement
    }
    assert(!v.neighbors.contains(v))
    assert(replacement.neighbors.isEmpty)
    assert(replacement.edges_to_me.isEmpty)
    replacement.neighbors ++= v.neighbors
    replacement.edges_to_me ++= v.edges_to_me
    replacement.accepting_neighbors = v.accepting_neighbors

    v.neighbors.clear()
    v.edges_to_me.clear()


    assert(this.nodes.contains(v))
    assert(this.exact.pointsSeen.contains(v))
    assert(!this.nodes.contains(replacement))
    assert(!this.exact.pointsSeen.contains(replacement))
    this.nodes -= v
    this.exact.pointsSeen -= v
    this.nodes += replacement
    this.exact.pointsSeen += replacement
    assert(!this.nodes.contains(v))
    assert(!this.exact.pointsSeen.contains(v))
    assert(this.nodes.contains(replacement))
    assert(this.exact.pointsSeen.contains(replacement))
  }

  def deleteNode(v: GrinchNode) = {
    v.edges_to_me.foreach{
      n =>
        n.neighbors -= v
        n.accepting_neighbors = (n.neighbors.size < n.grinch.config.max_degree)
    }
    v.neighbors.foreach{
      n =>
        n.edges_to_me -= v
    }
    v.neighbors.clear()
    v.edges_to_me.clear()
    assert(this.nodes.contains(v))
    assert(this.exact.pointsSeen.contains(v))
    this.nodes -= v
    this.exact.pointsSeen -= v
    assert(!this.nodes.contains(v))
    assert(!this.exact.pointsSeen.contains(v))
  }

  def _knn(v: GrinchNode, offlimits: mutable.Set[GrinchNode],k:Int, threadpool: ExecutorService = null , parByRoot: Boolean = true) = {
    var scores_and_nodes: IndexedSeq[(Float,GrinchNode)] = IndexedSeq[(Float,GrinchNode)]()
    val allowable_size = this.nodes.size - offlimits.size
    if (allowable_size == 0 || k * this.r * math.log(allowable_size) > allowable_size ||  exact_nn) {
      // println("running exact search!")
      // println(s"${new java.util.Date().toString} [_knn] Using Exact Search allowable.size = ${allowable_size}")
      scores_and_nodes = this.exact.cknn(v,k,offlimits,threadpool)
      scores_and_nodes
    } else {
      val knn = JavaHashSet[(Float,GrinchNode)]()
      var num_score_fn = 0.0
      val sim_cache = JavaHashMap[(GrinchNode,GrinchNode),Float]()
      var roots = JavaHashSet[GrinchNode]()
      if (allowable_size <= this.r || this.nodes.size < 10000) {
        roots ++= this.nodes.filterNot(offlimits.contains).take(this.r)
      } else {
        while (roots.size < this.r) {
          val trial = random.nextInt(this.nodes.size)
          if (!offlimits.contains(this.nodes(trial)))
            roots += this.nodes(trial)
        }
      }
      val path = JavaHashSet[GrinchNode]()
      if (parByRoot && threadpool != null) {
        Threading.parMap(roots,threadpool)(root => {
          val knn_res = root.cknn(v, k, offlimits, path, null).asScala
          knn_res
        }).foreach{
          case knn_res =>
            knn_res.foreach {
              pair =>
                knn += pair
            }
        }
      } else {
        roots.foreach {
          root =>
            val knn_res = root.cknn(v, k, offlimits, path, threadpool).asScala
            if (knn_res.nonEmpty) {
              knn_res.foreach {
                pair =>
                  knn += pair
              }
            }
        }
      }
      val scores_and_nodes = knn.toIndexedSeq
      scores_and_nodes.sortBy(a => (-a._1,a._2)).take(k)
    }
  }

  def cknn(v:GrinchNode,k:Int  ,offlimits: mutable.Set[GrinchNode], threadpool: ExecutorService = null) = {
    val scores_and_nodes = this._knn(v, offlimits, k,threadpool)
    val sorted = scores_and_nodes.sortBy(x => (-x._1,x._2))
    sorted
  }

  def cknn_and_insert(v:GrinchNode,k:Int,offlimits: mutable.Set[GrinchNode], threadpool: ExecutorService = null) = {
    if (this.nodes.isEmpty) {
      this.nodes += v
      this.exact.pointsSeen.add(v)
      IndexedSeq()
    } else {
      val scores_and_nodes = this._knn(v,offlimits,k,threadpool)
      var best_score: Float = Float.NegativeInfinity
      var best_node: GrinchNode = null
      scores_and_nodes.foreach {
        case (score, node) =>
          v.add_link(node)
          if (best_node == null || score > best_score || (score == best_score && best_node.id.compareTo(node.id) < 0)) {
            best_score = score
            best_node = node
          }
      }
      this.nodes += v
      this.exact.pointsSeen.add(v)
      val sorted = scores_and_nodes.sortBy(x => (-x._1,x._2))
      sorted
    }
  }

}
