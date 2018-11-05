// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_GRAPH_HPP
#define NETKET_GRAPH_HPP

#include "Utils/json_helper.hpp"
#include "abstract_graph.hpp"
#include "custom_graph.hpp"
#include "hypercube.hpp"

namespace netket {

// TODO(twesterhout): Move me to some .cpp file somewhere
std::unique_ptr<AbstractGraph> make_graph(const json& pars) {
  // Check if a graph is explicitely defined in the input
  if (FieldExists(pars, "Graph")) {
    // Checking if we are using a graph in the hard-coded library
    if (FieldExists(pars["Graph"], "Name")) {
      std::string graph_name = pars["Graph"]["Name"];
      if (graph_name == "Hypercube") {
        return make_hypercube(pars["Graph"]);
      } else if (graph_name == "Custom") {
        return make_custom_graph(pars["Graph"]);
      } else {
        std::stringstream s;
        s << "Unknown Graph type: " << graph_name;
        throw InvalidInputError(s.str());
      }
    }
    // Otherwise try with a user-defined graph
    else {
      return make_custom_graph(pars["Graph"]);
    }
  } else {
    if (FieldExists(pars, "Hilbert")) {
      int size = FieldVal<int>(pars["Hilbert"], "Size", "Graph");
      return netket::make_unique<CustomGraph>(size);
    } else {
      std::stringstream s;
      s << "Unknown Graph type";
      throw InvalidInputError(s.str());
    }
  }
}

#if 0
class Graph : public AbstractGraph {
  std::unique_ptr<AbstractGraph> g_;

 public:
  explicit Graph(const json& pars) {
    // Check if a graph is explicitely defined in the input
    if (FieldExists(pars, "Graph")) {
      // Checking if we are using a graph in the hard-coded library
      if (FieldExists(pars["Graph"], "Name")) {
        std::string graph_name = pars["Graph"]["Name"];
        if (graph_name == "Hypercube") {
          g_ = make_hypercube(pars["Graph"]);
        } else if (graph_name == "Custom") {
          g_ = make_custom_graph(pars["Graph"]);
        } else {
          std::stringstream s;
          s << "Unknown Graph type: " << graph_name;
          throw InvalidInputError(s.str());
        }
      }
      // Otherwise try with a user-defined graph
      else {
        g_ = make_custom_graph(pars["Graph"]);
      }
    } else {
      if (FieldExists(pars, "Hilbert")) {
        int size = FieldVal<int>(pars["Hilbert"], "Size", "Graph");
        g_ = netket::make_unique<CustomGraph>(size);
      } else {
        std::stringstream s;
        s << "Unknown Graph type";
        throw InvalidInputError(s.str());
      }
    }
  }

  /*
  explicit Graph(const std::string& name, const pybind11::kwargs& kwargs) {
    if (name == "Hypercube") {
      g_ = make_hypercube(kwargs);
    } else if (name == "Custom") {
      g_ = netket::make_unique<CustomGraph>(kwargs);
    } else {
      std::stringstream s;
      s << "Unknown Graph type: " << name;
      throw InvalidInputError(s.str());
    }
  }
  */

  int Nsites() const override { return g_->Nsites(); }

  int Size() const { return g_->Nsites(); }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return g_->AdjacencyList();
  }

  std::vector<std::vector<int>> SymmetryTable() const override {
    return g_->SymmetryTable();
  }

  const ColorMap& EdgeColors() const override { return g_->EdgeColors(); }

  template <typename Func>
  void BreadthFirstSearch(int start, int max_depth, Func visitor_func) const {
    g_->BreadthFirstSearch(start, max_depth, visitor_func);
  }

  template <typename Func>
  void BreadthFirstSearch(int start, Func visitor_func) const {
    BreadthFirstSearch(start, Nsites(), visitor_func);
  }

  template <typename Func>
  void BreadthFirstSearch(Func visitor_func) const {
    g_->BreadthFirstSearch(visitor_func);
  }

  bool IsBipartite() const override { return g_->IsBipartite(); }

  bool IsConnected() const override { return g_->IsConnected(); }

  std::vector<int> Distances(int root) const override {
    return g_->Distances(root);
  }

  std::vector<std::vector<int>> AllDistances() const override {
    return g_->AllDistances();
  }
};
#endif
}  // namespace netket

#endif
