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

import cc.factorie.util.{JavaHashSet, Threading}
import grinch.core.{GPQComparator, GrinchNode}

import scala.collection.mutable
import scala.collection.JavaConverters._

class Exact {

  val pointsSeen = JavaHashSet[GrinchNode]()

  def cknn(gn: GrinchNode, k: Int, offlimits: mutable.Set[GrinchNode],threadpool: ExecutorService= null) = {
    val distances = if (threadpool == null)
      pointsSeen.diff(offlimits).map(gn2 => (gn2.grinch.e_score(gn.ent,gn2.ent),gn2)) // minusNorm(gn.pt.value,gn2.pt.value),gn2))
    else
      Threading.parMap(pointsSeen.diff(offlimits),threadpool)(gn2 => {(gn2.grinch.e_score(gn.ent,gn2.ent),gn2) }).toIterable
    val pq = new java.util.PriorityQueue[(Float,GrinchNode)](GPQComparator)
    distances.foreach{
      case (d,pt) =>
        pq.add((d,pt))
        if (pq.size() > k) {
          pq.poll()
        }
    }
    pq.iterator().asScala.toIndexedSeq.sortBy(x => (-x._1,x._2))
  }

  def cknn_and_insert(point: GrinchNode, k: Int, offlimits: mutable.Set[GrinchNode],threadpool: ExecutorService = null) = {
    val res = cknn(point,k,offlimits,threadpool)
    pointsSeen.add(point)
    res
  }

}
