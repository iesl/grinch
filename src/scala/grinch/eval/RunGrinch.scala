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

package grinch.eval

import java.io.{File, PrintWriter}
import java.util.PriorityQueue

import cc.factorie.util.{DefaultCmdOptions, Threading}
import grinch.core._
import grinch.utils.ComparisonCounter
import grinch._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


trait RunGrinchOpts extends DefaultCmdOptions with GrinchConfigOpts {
  val input = new CmdOption[String]("input","The input file to cluster (Required).",true)
  val outDir = new CmdOption[String]("outdir","Where to write the experiment output (Required)",true)
  val algorithm = new CmdOption[String]("algorithm","Perch","STRING","The algorithm name to record in the results. Default: Perch")
  val datasetName = new CmdOption[String]("dataset","","STRING","The dataset name to record in the results. Default: input filename")
  val threads = new CmdOption[Int]("threads",4,"INT","The number of threads to use. Default: 24")
  val maxFrontierSizeBeforeParallelization = new CmdOption[Int]("max-frontier-par",50,"INT","The min points before invoking parallelization")
  val L = new CmdOption[String]("max-leaves", "None", "INT or None", "maximum number of leaves.  Default: None (no clustering extracted)")
  val K = new CmdOption[String]("clusters", "None", "INT or None", "The number of clusters. Default: None (no clustering extracted) ")
  val exactDistThreshold = new CmdOption[Int]("exact-dist-thres",10,"INT","The number of points to search using exact dist threshold.")
  val pickKMethod = new CmdOption[String]("pick-k","approxKM","STRING","the method used for picking k: pointCounter, maxD, minD, approxKM (default), globalKM, localKM")
  val beam = new CmdOption[String]("beam","None","STRING","The beam size or None to not use a beam. Default None")
  val countComparisons = new CmdOption[Boolean]("count-comparisons",false,"boolean", "Whether or not to count comparisons. Default: False")
  val quiet = new CmdOption[Boolean]("quiet",false,"boolean","Whether to skip printed status updates. Default: False")
  val linkage = new CmdOption[String]("linkage","average","string","average coslink")

}
/**
  * Load a file of points. The file format is one point per line as:
  * point_id \t gold_label \t vector (tab separated)
  */
object LoadPoints {
    def loadLine(line: String) = {
        val splt = line.split("\t")
        if (splt.length < 2)
            println(splt.mkString("\t"))
        val pid = splt(0)
        val label = splt(1)
        val vec = splt.drop(2).map(_.toFloat)
        Point(pid,label,vec)
      }

    def loadFile(file: File): Iterator[Point] = file.lines("UTF-8").map(loadLine)

    def loadFile(filename: String): Iterator[Point] = loadFile(new File(filename))
  }
/**
  * Run Grinch on a given dataset.
  */
object RunGrinch {

  def main(args: Array[String]): Unit = {

    // set random seed
    implicit val rdom = new Random(17)

    // Parse command line arguments
    val opts = new RunGrinchOpts {}
    opts.parse(args)

    println("Running EvalDataset")
    println("Command line arguments: ")
    opts.values.foreach(f => println(s"${f.name}: ${f.value}"))

    if (opts.countComparisons.value)
      ComparisonCounter.on()

    val outDir = new File(opts.outDir.value)
    outDir.mkdirs()

    var collapsibles = {
      if (opts.L.wasInvoked && opts.L.value.toLowerCase != "none" && opts.L.value.toInt > 0)
        new PriorityQueue[(Float, GrinchNode)](GPQComparator)
      else
        null
    }

    val L = {
      if (opts.L.wasInvoked && opts.L.value.toLowerCase != "none" && opts.L.value.toInt > 0) {
        opts.L.value.toInt
      } else {
        -1
      }
    }

    println(s"Collapsibles: ${collapsibles}")
    println(s"L: $L")

    val beQuiet = opts.quiet.value
    val threads = opts.threads.value
    val maxFrontierSizeBeforeParallelization = if (opts.maxFrontierSizeBeforeParallelization.wasInvoked) opts.maxFrontierSizeBeforeParallelization.value else opts.threads.value

    val grinchConfig = Config(opts)
    println("GRINCH CONFIG:")
    println(grinchConfig)

    val threadpool = if (opts.threads.value > 1) Threading.newFixedThreadPool(opts.threads.value) else null
    val grinch = if (opts.linkage.value == "average") new ParAvgLinkGrinch(grinchConfig,threadpool) else new ParCosLinkGrinch(grinchConfig,threadpool)
    val points = LoadPoints.loadFile(opts.input.value)

    val start = System.currentTimeMillis()

    grinch.build_dendrogram(points)

    val root = grinch.root
    if (threadpool != null)
      threadpool.shutdown()
    val end = System.currentTimeMillis()
    val runningTimeSeconds = (end - start).toFloat / 1000.0

    // Write running time to a file
    val pw = new PrintWriter(new File(outDir,"running_time.txt"))
    val datasetName = if (opts.datasetName.wasInvoked) opts.datasetName.value else new File(opts.input.value).getName
    pw.println(s"${opts.algorithm.value}\t$datasetName\t$runningTimeSeconds")
    pw.close()

    // Write tree to the file
    root.serializeTree(new File(outDir,"tree.tsv"))

    // Print the number of comparisons
    val numberOfComparisons = ComparisonCounter.count
    val compPW = new PrintWriter(new File(outDir,"comparisons.txt"))
    compPW.println(s"${opts.algorithm.value}\t$datasetName\t${numberOfComparisons.get()}")
    compPW.close()
  }


}