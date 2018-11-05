#ifndef NETKET_ODESOLVER_HPP
#define NETKET_ODESOLVER_HPP

#include <memory>

#include "Graph/graph.hpp"
#include "Hamiltonian/MatrixWrapper/matrix_wrapper.hpp"
#include "Hamiltonian/hamiltonian.hpp"
#include "Utils/all_utils.hpp"

#include "TimeStepper/time_stepper.hpp"

namespace netket {

// The real-time evolution code is work-in-progress and likely to change in the
// future.

class TimeEvolutionDriver {
 public:
  using VectorType = Eigen::VectorXcd;
  using MatrixType = AbstractMatrixWrapper<Hamiltonian>;
  using StepperType = ode::AbstractTimeStepper<VectorType>;

  static TimeEvolutionDriver FromJson(const json& pars) {
    auto graph = make_graph(pars);
    Hilbert hilbert(*graph, pars);
    Hamiltonian hamiltonian(hilbert, pars);

    auto pars_te = pars["TimeEvolution"];
    auto matrix = ConstructMatrixWrapper(pars_te, hamiltonian);
    auto stepper =
        ode::ConstructTimeStepper<VectorType>(pars_te, matrix->GetDimension());

    auto range = ode::TimeRange::FromJson(pars_te);

    return TimeEvolutionDriver(std::move(matrix), std::move(stepper), range);
  }

  TimeEvolutionDriver(
      std::unique_ptr<MatrixType> matrix, std::unique_ptr<StepperType> stepper,
      const ode::TimeRange& range)  // const reference to make Codacy happy
      : hmat_(std::move(matrix)), stepper_(std::move(stepper)), range_(range) {
    ode_system_ = [this](const VectorType& x, VectorType& dxdt, double /*t*/) {
      static const std::complex<double> im{0., 1.};
      dxdt.noalias() = -im * hmat_->Apply(x);
    };
  }

  void Run(VectorType& state, ode::ObserverFunction<VectorType> observer_func) {
    assert(state.size() == GetDimension());

    ode::Integrate(*stepper_, ode_system_, state, range_, observer_func);
  }

  int GetDimension() const { return hmat_->GetDimension(); }

 private:
  std::unique_ptr<MatrixType> hmat_;
  std::unique_ptr<StepperType> stepper_;
  ode::OdeSystemFunction<VectorType> ode_system_;

  ode::TimeRange range_;
};

void RunTimeEvolution(const json& pars) {
  auto driver = TimeEvolutionDriver::FromJson(pars);

  auto pars_te = FieldVal(pars, "TimeEvolution");
  auto confs = FieldVal(pars_te, "InitialStates");
  if (confs.size() == 0) {
    std::cerr << "No configurations specified for time evolution" << std::endl;
    std::exit(0);
  }

  std::string filename_template = pars_te["OutputFiles"].get<std::string>();
  if (filename_template.size() == 0) {
    std::cerr << "Configuration with empty OutputFiles template" << std::endl;
    std::exit(1);
  }
  static const std::string marker("%i");
  auto pos = filename_template.find(marker);
  if (pos == filename_template.npos) {
    std::cerr << "OutputFiles is lacking '%i'" << std::endl;
    std::exit(1);
  }

  int mpi_rank_;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
  int mpi_size_;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

  for (size_t i = mpi_rank_; i < confs.size(); i += mpi_size_) {
    std::stringstream index_str;
    index_str << i;
    std::string filename(filename_template);
    filename.replace(pos, marker.size(), index_str.str());
    std::ofstream stream(filename);

    auto observer_func = [&](const TimeEvolutionDriver::VectorType& x,
                             double t) {
      json out;
      out["Time"] = t;
      out["State"] = x;
      stream << out << std::endl;
    };

    auto conf = confs[i];
    using InitialStateType = std::vector<std::complex<double>>;
    auto initial_vec = conf.get<InitialStateType>();

    if (static_cast<std::ptrdiff_t>(initial_vec.size()) !=
        driver.GetDimension()) {
      std::cout << "Error: Initial states need to have "
                << driver.GetDimension() << " entries." << std::endl;
      std::abort();
    }

    TimeEvolutionDriver::VectorType initial =
        Eigen::Map<TimeEvolutionDriver::VectorType>(initial_vec.data(),
                                                    initial_vec.size());

    driver.Run(initial, observer_func);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

}  // namespace netket

#endif  // NETKET_ODESOLVER_HPP
