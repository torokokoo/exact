# Time Series — Inner Workings

## Class Hierarchy

```
TimeSeriesSets          ← top-level manager (one per run)
  └── vector<TimeSeriesSet*> time_series
        └── map<string, TimeSeries*>   ← one TimeSeries per column
```

---

## `TimeSeries` — Single Column

```cpp
TimeSeries(string name)
void add_value(double)
void calculate_statistics()     // computes min, max, average, std_dev, variance,
                                //         min_change, max_change
void normalize_min_max(double min, double max)       // x = (x-min)/(max-min)
void normalize_avg_std_dev(double avg, double std_dev, double norm_max)
                                                     // x = ((x-avg)/std_dev)/norm_max
double get_correlation(const TimeSeries* other, int lag) // Pearson correlation
void cut(int start, int stop)    // trim values, recalculate statistics
TimeSeries* copy()
void copy_values(vector<double>& out)
```

---

## `TimeSeriesSet` — One CSV File

Loads a single CSV file at construction time. First row = column headers.

```cpp
TimeSeriesSet(string filename, const vector<string>& fields)
```

Key methods:
```cpp
void export_time_series(vector<vector<double>>& out)
void export_time_series(vector<vector<double>>& out,
                        const vector<string>& fields,
                        const vector<string>& shift_fields,
                        int32_t time_offset)
```

**Time offset export logic:**
- For `shift_fields` with `time_offset > 0`: the output at row `i` is the value at row `i + time_offset` — i.e. the last `time_offset` rows are excluded from inputs, and the first `time_offset` rows are excluded from outputs.
- `time_offset = 0` exports input and output from the same row (useful for autoencoders).

```cpp
void split(int32_t slices, vector<TimeSeriesSet*>& out)  // chop into N temporal chunks
void cut(int start, int stop)
void select_parameters(vector<string>& names)            // keep only listed columns
```

---

## `TimeSeriesSets` — Full Dataset Manager

### Construction from arguments
```cpp
TimeSeriesSets* TimeSeriesSets::generate_from_arguments(const vector<string>& arguments)
```

Reads:
- `--training_filenames <files>+` and `--validation_filenames <files>+`
  OR `--filenames <files>+` with optional `--training_indexes` / `--test_indexes`
- `--input_parameter_names <cols>+` and `--output_parameter_names <cols>+`
  OR `--parameters <name> <setting> [<min> <max>]+`
- `--normalize <none|min_max|avg_std_dev>` (default: none)
- `--shift_parameter_names <cols>+` (optional output-column time-shift)
- `--time_offset <int>` (how many rows into the future to predict)

### Normalization

Normalization statistics are computed **from training data only** and then applied to both training and validation sets. The stored bounds (`normalize_mins`, `normalize_maxs`, `normalize_avgs`, `normalize_std_devs`) are embedded in every serialized genome so that predictions can be denormalized at inference time.

```cpp
void normalize_min_max()
void normalize_avg_std_dev()
double denormalize(string field_name, double value)
```

### Data export
```cpp
void export_training_series(int32_t time_offset,
                            vector<vector<vector<double>>>& inputs,   // [series][time][dim]
                            vector<vector<vector<double>>>& outputs)
void export_test_series(...)    // same shape for validation/test
void export_series_by_name(string field, vector<vector<double>>& out) // single column, all files
```

### Splitting for cross-validation
```cpp
void split_series(int32_t series_index, int32_t number_slices)  // one file → N chunks
void split_all(int32_t number_slices)                           // all files → N chunks each
void set_training_indexes(const vector<int>& idxs)
void set_test_indexes(const vector<int>& idxs)
```

---

## Data Flow in a Typical Run

```
TimeSeriesSets::generate_from_arguments()
    │
    ├── load each CSV into TimeSeriesSet
    ├── compute normalization stats from training_indexes
    └── apply normalization to all series

get_train_validation_data()          (process_arguments.cxx)
    │
    ├── export_training_series()  →  training_inputs[s][t][i], training_outputs[s][t][o]
    ├── export_test_series()      →  validation_inputs[s][t][i], validation_outputs[s][t][o]
    └── optionally slice_input_data() into fixed-length windows
              (--train_sequence_length splits each series into chunks of that length)

genome->backpropagate_stochastic(training_inputs, training_outputs,
                                  validation_inputs, validation_outputs, ...)
```

The 3D vectors `[series][timestep][dimension]` are the primary data format passed between the time-series layer and the RNN training layer.
