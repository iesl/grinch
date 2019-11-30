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

import cc.factorie.util.{CmdOptions, JavaHashSet, Threading}

import scala.collection.mutable.ArrayBuffer
import grinch._
import grinch.nn.{Exact, NSW}
import grinch.utils.{ComparisonCounter, ConsistentId}

import scala.collection.{GenSet, mutable}
import scala.collection.JavaConverters._

/**
  * Statistics stored for a subtree to compute the linkage function.
  * @param dim - dimensionality of the data
  * @param initGnode - The grinch node at the root of the subtree
  */
class Ent(dim: Int, initGnode: GrinchNode = null) extends Comparable[Ent] {

  override def compareTo(o: Ent): Int = this.hashCode() - o.hashCode()

  val d = dim
  var sum: Array[Float] = null
  var centroid: Array[Float] = null
  var numPts = 0
  var points = new ArrayBuffer[Point]()
  var needsUpdate = false
  var gnode: GrinchNode = initGnode

  def zero() = {
    sum = Array.fill[Float](dim)(0.0f)
    centroid = Array.fill[Float](dim)(0.0f)
    numPts = 0
    points = new ArrayBuffer[Point]()
  }

  def observe(pt: Point) = {
    zero()
    this.sum += pt.value
    this.numPts += 1
    this.centroid = this.sum / numPts
    this.points.append(pt)
  }

  def update() = {
    val toUpdate = new ArrayBuffer[Ent]()
    val toCheck = new java.util.LinkedList[Ent]()
    toCheck.add(this)
    while (!toCheck.isEmpty) {
      val curr = toCheck.pop()
      if (curr.needsUpdate) {
        toUpdate.append(curr)
        curr.gnode.children.foreach {
          c => toCheck.add(c.ent)
        }
      }
    }
    toUpdate.reverseIterator.foreach(_.singleUpdate())
  }

  def singleUpdate() = {
    assert(this.needsUpdate)
    this.needsUpdate = false
    val c1 = this.gnode.children(0).ent
    val c2 = this.gnode.children(1).ent
    assert(!c1.needsUpdate)
    assert(!c2.needsUpdate)
    zero()
    this.sum += c1.sum
    this.sum += c2.sum
    this.numPts = c1.numPts + c2.numPts
    this.centroid = this.sum / this.numPts
    if (this.gnode == null || this.gnode.grinch == null || !this.gnode.grinch.can_forget_points) {
      this.points ++= c1.points
      this.points ++= c2.points
    }
  }

  def copy() = {
    val newEnt = new Ent(this.d)
    newEnt.needsUpdate = true
    newEnt
  }

  def mergedRep(other: Ent) = {
    this.needsUpdate = true
    this
  }

}