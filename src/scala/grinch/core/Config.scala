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


import cc.factorie.util.CmdOptions


trait GrinchConfigOpts extends CmdOptions {
  val k = new CmdOption[Int]("k",25,"","")
  val max_num_leaves = new CmdOption[Int]("max_num_leaves",-1,"","")
  val graft_beam = new CmdOption[Int]("graft_beam",100000000,"","")
  val rotation_size_cap = new CmdOption[Int]("rotation_size_cap",100000000,"","")
  val graft_size_cap = new CmdOption[Int]("graft_size_cap",100000000,"","")
  val restruct_size_cap = new CmdOption[Int]("restruct_size_cap",100000000,"","")
  val perform_rotate = new CmdOption[Boolean]("perform_rotate",true,"","")
  val perform_graft = new CmdOption[Boolean]("perform_graft",true,"","")
  val perform_restruct = new CmdOption[Boolean]("perform_restruct",true,"","")
  val single_graft_search = new CmdOption[Boolean]("single_graft_search",false,"","")
  val exact_dist_thresh = new CmdOption[Int]("exact_dist_thresh",100,"","")
  val single_elimination = new CmdOption[Boolean]("single_elimination",false,"","")
  val graft_debug = new CmdOption[Boolean]("graft_debug",false,"","")
  val max_degree = new CmdOption[Int]("max_degree",100,"","")
  val nsw_r = new CmdOption[Int]("nsw_r",3,"","")
  val exact_nn = new CmdOption[Boolean]("exact_nn",false,"","")
  val max_search_time = new CmdOption[Int]("max_search_time",Int.MaxValue,"","")
  val skip_graft_at = new CmdOption[Int]("skip_graft_at",-1,"","")
  val nsw_debug = new CmdOption[Boolean]("nsw_debug",false,"","")
  val crazy_approx = new CmdOption[Boolean]("crazy_approx",false,"","")

}

case class Config(k: Int = 25,
                  max_num_leaves: Int = -1,
                  graft_beam: Int = 100000000,
                  rotation_size_cap: Int = 100000000,
                  graft_size_cap: Int = 100000000,
                  restruct_size_cap: Int = 100000000,
                  perform_graft: Boolean = true,
                  perform_restruct: Boolean = true,
                  perform_rotate: Boolean = true,
                  single_graft_search: Boolean = false,
                  exact_dist_thresh: Int = 100,
                  single_elimination: Boolean = false,
                  graft_debug: Boolean = false,
                  max_degree: Int = 100,
                  nsw_r: Int = 2,
                  exact_nn: Boolean = false,
                  maxSearchTime: Int = Int.MaxValue,
                  nsw_debug: Boolean = false,
                  skip_graft_at: Int = -1,use_crazy_approx: Boolean = false) {}

object Config {
  def apply(opts:GrinchConfigOpts): Config = {
    Config(opts.k.value,
      opts.max_num_leaves.value,
      opts.graft_beam.value,
      opts.rotation_size_cap.value,
      opts.graft_size_cap.value,
      opts.restruct_size_cap.value,
      opts.perform_graft.value,
      opts.perform_restruct.value,
      opts.perform_rotate.value,
      opts.single_graft_search.value,
      opts.exact_dist_thresh.value,
      opts.single_elimination.value,
      opts.graft_debug.value,
      opts.max_degree.value,
      opts.nsw_r.value,
      opts.exact_nn.value,
      opts.max_search_time.value,
      opts.nsw_debug.value,
      opts.skip_graft_at.value)
  }
}