#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-coal}"            # coal | wind | c172
NP="${NP:-4}"                         # MPI processes
MAX_GENOMES="${MAX_GENOMES:-2000}"
BP_EPOCHS="${BP_EPOCHS:-20}"
NUM_MUTATIONS="${NUM_MUTATIONS:-2}"   # repo base_run scripts use 2; use 1 for one op per mutated child
WEIGHT_INIT="${WEIGHT_INIT:-xavier}"  # xavier = X-L-L; kaiming = K-L-L
REPO_PARENT="${REPO_PARENT:-$PWD}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"     # set 0 to skip dependency installation
INSTALL_OPTIONAL_DEPS="${INSTALL_OPTIONAL_DEPS:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"         # set 1 to reuse an existing build
DRY_RUN="${DRY_RUN:-0}"               # set 1 to print the command without running it
BUILD_DIR="${BUILD_DIR:-build_mpi_lyu2021}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${REPO_DIR:-}" ]]; then
  REPO_DIR="$REPO_DIR"
elif [[ -f "$SCRIPT_DIR/CMakeLists.txt" && -d "$SCRIPT_DIR/.git" ]]; then
  REPO_DIR="$SCRIPT_DIR"
else
  REPO_DIR="$REPO_PARENT/exact"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"

OS_NAME="$(uname -s)"

if [[ "$INSTALL_DEPS" == "1" && "$OS_NAME" == "Linux" ]] && command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    build-essential cmake git pkg-config \
    openmpi-bin libopenmpi-dev \
    default-libmysqlclient-dev libtiff-dev libpng-dev \
    graphviz clang-format
fi

if [[ "$INSTALL_DEPS" == "1" && "$OS_NAME" == "Darwin" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew no está instalado. Instala Homebrew o ejecuta con INSTALL_DEPS=0 y OpenMPI en PATH." >&2
    exit 4
  fi

  BREW_PACKAGES=()
  if ! command -v cmake >/dev/null 2>&1; then
    BREW_PACKAGES+=(cmake)
  fi
  if ! command -v mpirun >/dev/null 2>&1 || ! command -v mpicxx >/dev/null 2>&1; then
    BREW_PACKAGES+=(open-mpi)
  fi
  if [[ "$INSTALL_OPTIONAL_DEPS" == "1" ]]; then
    for pkg in libtiff libpng graphviz clang-format; do
      if ! brew list --formula "$pkg" >/dev/null 2>&1; then
        BREW_PACKAGES+=("$pkg")
      fi
    done
  fi

  if ((${#BREW_PACKAGES[@]})); then
    brew install "${BREW_PACKAGES[@]}"
  fi
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone https://github.com/travisdesell/exact.git "$REPO_DIR"
fi

cd "$REPO_DIR"

if [[ "$OS_NAME" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
  BREW_PREFIX="$(brew --prefix)"
  for mpi_prefix in "$(brew --prefix open-mpi 2>/dev/null || true)" "$BREW_PREFIX/opt/open-mpi"; do
    if [[ -n "$mpi_prefix" && -d "$mpi_prefix/bin" ]]; then
      export PATH="$mpi_prefix/bin:$PATH"
      export CMAKE_PREFIX_PATH="$mpi_prefix:${CMAKE_PREFIX_PATH:-}"
      break
    fi
  done
fi

if ! command -v mpirun >/dev/null 2>&1 || ! command -v mpicxx >/dev/null 2>&1; then
  echo "No encontré mpirun/mpicxx. En macOS instala OpenMPI con: brew install open-mpi" >&2
  exit 5
fi

export CC="${CC:-$(command -v mpicc)}"
export CXX="${CXX:-$(command -v mpicxx)}"
export ASAN_OPTIONS="${ASAN_OPTIONS:-detect_leaks=0}"

if [[ "$SKIP_BUILD" != "1" ]]; then
  cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release

  if command -v nproc >/dev/null 2>&1; then
    BUILD_JOBS="$(nproc)"
  elif command -v getconf >/dev/null 2>&1; then
    BUILD_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
  elif command -v sysctl >/dev/null 2>&1; then
    BUILD_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
  else
    BUILD_JOBS="4"
  fi

  cmake --build "$BUILD_DIR" -j"$BUILD_JOBS"
fi

cd "$BUILD_DIR"

case "$DATASET" in
  coal)
    INPUT_PARAMETERS=(
      Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate
      Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total
      Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split
      Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int
    )
    OUTPUT_PARAMETERS=(Main_Flm_Int)
    TRAIN_FILES=(../datasets/2018_coal/burner_[0-9].csv)
    VALID_FILES=(../datasets/2018_coal/burner_1[0-1].csv)
    NORMALIZE_ARGS=()
    SEQ_ARGS=(--train_sequence_length 50)
    ;;
  wind)
    INPUT_PARAMETERS=(
      Ba_avg Rt_avg DCs_avg Cm_avg P_avg S_avg Cosphi_avg Db1t_avg Db2t_avg
      Dst_avg Gb1t_avg Gb2t_avg Git_avg Gost_avg Ya_avg Yt_avg Ws_avg Wa_avg
      Ot_avg Nf_avg Nu_avg Rbt_avg
    )
    OUTPUT_PARAMETERS=(P_avg)
    TRAIN_FILES=(
      ../datasets/2020_wind_engine/turbine_R80711_2017-2020_[1-9].csv
      ../datasets/2020_wind_engine/turbine_R80711_2017-2020_1[0-9].csv
      ../datasets/2020_wind_engine/turbine_R80711_2017-2020_2[0-4].csv
    )
    VALID_FILES=(
      ../datasets/2020_wind_engine/turbine_R80711_2017-2020_2[5-9].csv
      ../datasets/2020_wind_engine/turbine_R80711_2017-2020_3[0-1].csv
    )
    NORMALIZE_ARGS=(--normalize min_max)
    SEQ_ARGS=()
    ;;
  c172)
    INPUT_PARAMETERS=(
      AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4
      E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM
      FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG
      WndDr WndSpd
    )
    OUTPUT_PARAMETERS=(Pitch)
    TRAIN_FILES=(../datasets/2019_ngafid_transfer/c172_file_[1-9].csv)
    VALID_FILES=(../datasets/2019_ngafid_transfer/c172_file_1[0-2].csv)
    NORMALIZE_ARGS=(--normalize min_max)
    SEQ_ARGS=()
    ;;
  *)
    echo "DATASET debe ser: coal, wind o c172" >&2
    exit 2
    ;;
esac

OUT_DIR="../test_output/${DATASET}_mpi_lyu2021_${RUN_ID}"
mkdir -p "$OUT_DIR"

RUN_CMD=(
  mpirun -np "$NP" ./mpi/examm_mpi
  --training_filenames "${TRAIN_FILES[@]}"
  --validation_filenames "${VALID_FILES[@]}"
  --time_offset 1
  --input_parameter_names "${INPUT_PARAMETERS[@]}"
  --output_parameter_names "${OUTPUT_PARAMETERS[@]}"
  --number_islands 10
  --island_size 10
  --max_genomes "$MAX_GENOMES"
  --bp_iterations "$BP_EPOCHS"
  --num_mutations "$NUM_MUTATIONS"
  --possible_node_types simple UGRNN MGU GRU delta LSTM
  --weight_initialize "$WEIGHT_INIT"
  --weight_inheritance lamarckian
  --mutated_component_weight lamarckian
  --weight_update adagrad
  --eps 0.000001
  --beta1 0.99
  --generate_op_log 1
  --genome_size_log 1
  --save_genome_option the_best
  --output_directory "$OUT_DIR"
  --std_message_level INFO
  --file_message_level INFO
)

if ((${#NORMALIZE_ARGS[@]})); then
  RUN_CMD+=("${NORMALIZE_ARGS[@]}")
fi

if ((${#SEQ_ARGS[@]})); then
  RUN_CMD+=("${SEQ_ARGS[@]}")
fi

printf 'Running dataset: %s\n' "$DATASET"
printf 'Output directory: %s\n' "$OUT_DIR"
printf 'Command:'
printf ' %q' "${RUN_CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

"${RUN_CMD[@]}" 2>&1 | tee "$OUT_DIR/run_stdout_stderr.log"

FITNESS_LOG="$OUT_DIR/fitness_log.csv"
SUMMARY="$OUT_DIR/mse_summary.csv"

if [[ -s "$FITNESS_LOG" ]]; then
  BEST_MSE="$(
    awk -F, 'NR > 1 && $5 != "" {m=$5} END {gsub(/^[ \t]+|[ \t]+$/, "", m); print m}' "$FITNESS_LOG"
  )"
  {
    echo "dataset,best_validation_mse,fitness_log,output_dir"
    echo "$DATASET,$BEST_MSE,$FITNESS_LOG,$OUT_DIR"
  } > "$SUMMARY"
  echo "Best validation MSE: $BEST_MSE"
  echo "Resumen MSE: $SUMMARY"
else
  echo "No se encontró fitness_log.csv en $OUT_DIR" >&2
  exit 3
fi
