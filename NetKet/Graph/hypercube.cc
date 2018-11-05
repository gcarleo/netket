
#include "Graph/hypercube.hpp"

namespace netket {

namespace {
  static void CheckArgs(int const L, int const ndim, bool const pbc)
  {
    if (L <= 0) {
      std::ostringstream msg;
      msg << "Side length must be at least 1, but got " << L;
      throw InvalidInputError{msg.str()};
    }
    if (ndim <= 0) {
      std::ostringstream msg;
      msg << "Dimension must be at least 1, but got " << ndim;
      throw InvalidInputError{msg.str()};
    }
    if (pbc && L <= 2) {
      throw InvalidInputError(
          "L<=2 hypercubes cannot have periodic boundary conditions");
    }
  }
} // namespace

Hypercube::Hypercube(int const L, int const ndim, bool const pbc)
    : L_{L},
      ndim_{ndim},
      pbc_{pbc},
      sites_{},
      coord2sites_{},
      adjlist_{},
      eclist_{},
      nsites_{0} {
  CheckArgs(L, ndim, pbc);
  Init(nullptr);
}

Hypercube::Hypercube(int const L, int const ndim, bool const pbc,
                     std::vector<std::vector<int>> const &colorlist)
    : L_{L},
      ndim_{ndim},
      pbc_{pbc},
      sites_{},
      coord2sites_{},
      adjlist_{},
      eclist_{},
      nsites_{0} {
  CheckArgs(L, ndim, pbc);
  Init(&colorlist);
}

void Hypercube::Init(std::vector<std::vector<int>> const *colorlist) {
  assert(L_ > 0 && "Bug! L_>=1 by construction.");
  assert(ndim_ >= 1 && "Bug! ndim_>=1 by construction.");
  GenerateLatticePoints();
  GenerateAdjacencyList();
  // If edge colors are specificied read them in, otherwise set them all to 0
  if (colorlist != nullptr) {
    EdgeColorsFromList(*colorlist, eclist_);
  } else {
    InfoMessage() << "No colors specified, edge colors set to 0 " << '\n';
    EdgeColorsFromAdj(adjlist_, eclist_);
  }
  InfoMessage() << "Hypercube created\n";
  InfoMessage() << "Dimension = " << ndim_ << '\n';
  InfoMessage() << "L = " << L_ << '\n';
  InfoMessage() << "Pbc = " << pbc_ << '\n';
}

void Hypercube::GenerateLatticePoints() {
  std::vector<int> coord(ndim_, 0);

  nsites_ = 0;
  do {
    sites_.push_back(coord);
    coord2sites_[coord] = nsites_;
    nsites_++;
  } while (netket::next_variation(coord.begin(), coord.end(), L_ - 1));
}

void Hypercube::GenerateAdjacencyList() {
  adjlist_.resize(nsites_);

  for (int i = 0; i < nsites_; i++) {
    std::vector<int> neigh(ndim_);
    std::vector<int> neigh2(ndim_);

    neigh = sites_[i];
    neigh2 = sites_[i];
    for (int d = 0; d < ndim_; d++) {
      if (pbc_) {
        neigh[d] = (sites_[i][d] + 1) % L_;
        neigh2[d] = ((sites_[i][d] - 1) % L_ + L_) % L_;
        int neigh_site = coord2sites_.at(neigh);
        int neigh_site2 = coord2sites_.at(neigh2);
        adjlist_[i].push_back(neigh_site);
        adjlist_[i].push_back(neigh_site2);
      } else {
        if ((sites_[i][d] + 1) < L_) {
          neigh[d] = (sites_[i][d] + 1);
          int neigh_site = coord2sites_.at(neigh);
          adjlist_[i].push_back(neigh_site);
          adjlist_[neigh_site].push_back(i);
        }
      }

      neigh[d] = sites_[i][d];
      neigh2[d] = sites_[i][d];
    }
  }
}

// Returns a list of permuted sites equivalent with respect to
// translation symmetry
std::vector<std::vector<int>> Hypercube::SymmetryTable() const override {
  if (!pbc_) {
    throw InvalidInputError(
        "Cannot generate translation symmetries "
        "in the hypercube without PBC");
  }

  std::vector<std::vector<int>> permtable;

  std::vector<int> transl_sites(nsites_);
  std::vector<int> ts(ndim_);

  for (int i = 0; i < nsites_; i++) {
    for (int p = 0; p < nsites_; p++) {
      for (int d = 0; d < ndim_; d++) {
        ts[d] = (sites_[i][d] + sites_[p][d]) % L_;
      }
      transl_sites[p] = coord2sites_.at(ts);
    }
    permtable.push_back(transl_sites);
  }
  return permtable;
}

} // namespace netket

class Hypercube : public AbstractGraph {
  // edge of the hypercube
  const int L_;

  // number of dimensions
  const int ndim_;

  // whether to use periodic boundary conditions
  const bool pbc_;

  // contains sites coordinates
  std::vector<std::vector<int>> sites_;

  // maps coordinates to site number
  std::map<std::vector<int>, int> coord2sites_;

  // adjacency list
  std::vector<std::vector<int>> adjlist_;

  // Edge colors
  ColorMap eclist_;

  int nsites_;

 private:

 public:

  template <class Ptype>
  Hypercube(const Ptype &pars)
      : L_(FieldVal<int>(pars, "L", "Graph")),
        ndim_(FieldVal<int>(pars, "Dimension", "Graph")),
        pbc_(FieldOrDefaultVal(pars, "Pbc", true)) {
    if (pbc_ && L_ <= 2) {
      throw InvalidInputError(
          "L<=2 hypercubes cannot have periodic boundary conditions");
    }
    Init(pars);
  }


  template <class Ptype>
  void Init(const Ptype &pars) {
    assert(L_ > 0);
    assert(ndim_ >= 1);
    GenerateLatticePoints();
    GenerateAdjacencyList();

    // If edge colors are specificied read them in, otherwise set them all to
    // 0
    if (FieldExists(pars, "EdgeColors")) {
      std::vector<std::vector<int>> colorlist =
          FieldVal<std::vector<std::vector<int>>>(pars, "EdgeColors", "Graph");
      EdgeColorsFromList(colorlist, eclist_);
    } else {
      InfoMessage() << "No colors specified, edge colors set to 0 "
                    << std::endl;
      EdgeColorsFromAdj(adjlist_, eclist_);
    }

    InfoMessage() << "Hypercube created " << std::endl;
    InfoMessage() << "Dimension = " << ndim_ << std::endl;
    InfoMessage() << "L = " << L_ << std::endl;
    InfoMessage() << "Pbc = " << pbc_ << std::endl;
  }



  int Nsites() const override { return nsites_; }

  int Length() const { return L_; }

  int Ndim() const { return ndim_; }

  std::vector<std::vector<int>> Sites() const { return sites_; }

  std::vector<int> SiteCoord(int i) const { return sites_[i]; }

  std::vector<std::vector<int>> AdjacencyList() const override {
    return adjlist_;
  }

  std::map<std::vector<int>, int> Coord2Site() const { return coord2sites_; }

  int Coord2Site(const std::vector<int> &coord) const {
    return coord2sites_.at(coord);
  }

  bool IsBipartite() const override { return true; }

  bool IsConnected() const override { return true; }

  // Returns map of the edge and its respective color
  const ColorMap &EdgeColors() const override { return eclist_; }
};
