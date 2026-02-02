# LATTEBench

LATTEBench is an  LLM-powered
AuTomated Tabular feature Engineering (LATTE) framework, integrating multiple feature generation methods.

## Environment Setup

### 1. Install Dependencies

Create environment using Conda:

```bash
conda create --name llm4fe --file requirements.txt
conda activate llm4fe
```

Or install core dependencies using pip:

```bash
pip install openai pandas numpy scikit-learn torch transformers sentence-transformers \
    autogluon xgboost lightgbm autofeat zss python-graphviz
```

### 2. Configure API Keys

This project supports multiple LLM backends. Configure the corresponding API key based on the model you use.

Create a `.env` file or set environment variables:

```bash
# DeepSeek API (for deepseek-chat, deepseek-reasoner models)
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# OpenRouter API (for llama-3.1-8b-instruct, llama-3.1-70b-instruct models)
export OPENROUTER_API_KEY="your-openrouter-api-key"

# OpenAI API (for gpt-4o, gpt-4o-mini, etc.)
export OPENAI_API_KEY="your-openai-api-key"
```

**How to obtain API keys:**
- DeepSeek: https://platform.deepseek.com/
- OpenRouter: https://openrouter.ai/
- OpenAI: https://platform.openai.com/

## Quick Start

```bash
# Using CoT method
python latte.py --method CoT --data_name credit-g --llm_model gpt-4o

# Run benchmark tests
python bench.py --methods CoT Critic --datasets credit-g --seeds 1 2 3
```

---

## latte.py - Unified Method Entry Point

`latte.py` is the unified entry point of the framework. Use the `--method` parameter to specify different feature engineering methods.

### Supported Methods

| Method | Prompting | Features |
|--------|-------------|----------|
| `CoT` | Chain-of-Thought | Basic chain-of-thought, supports Positive-Negative feedback (--history) and Top-k feedback (--top) |
| `Critic` | Generator-Critic | Invokes a Critic Agent to provide improvement suggestions after feature generation |
| `OPRO` | Optimization by Prompting | Multi-turn dialogue within each iteration for progressive improvement |
| `OPROc` | OPRO with CART reasoning | Dialogue-based feature generation using decision tree generated rules |
| `ECoT` | EvoPrompt | Uses island model evolution, directly executes postfix expressions |
| `Evo` | EvoPrompt | Island model evolution, supports multiple output formats |
| `ToT` | Tree of Thought | Use Monte Carlo Tree Search to explore the feature space |

### Command-Line Arguments

#### Basic Arguments

```bash
--method         # Method selection: CoT, Critic, OPRO, OPROc, ECoT, Evo, ToT
--data_name      # Dataset name (default: credit-g)
--llm_model      # LLM model (default: deepseek-chat)
--output_format  # Output format: NL, cRPN, Code, Rule (default: NL)
--iter           # Number of iterations (default: 50)
--seed           # Random seed (default: 2)
--task_type      # Task type: 1=classification, 0=regression (default: 1)
```

#### CoT-Specific Arguments

```bash
--history        # Use history feedback: 1=yes, 0=no (default: 1)
--top            # Use Top-k history feedback: 1=yes, 0=no (default: 0)
```

#### OPRO/OPROc-Specific Arguments

```bash
--dialogue_turns # Number of dialogue turns per iteration (default: 10)
```

#### ECoT/Evo-Specific Arguments

```bash
--ind_num        # Number of individuals per island (default: 90)
--remove_time    # Frequency of removing weak individuals (default: 1)
--update_time    # Inter-island update frequency (default: 2)
--random_sample  # Random sampling (ECoT): 1=yes, 0=no (default: 0)
```

#### ToT-Specific Arguments

```bash
--num_thoughts       # Number of thoughts to generate (default: 2)
--max_steps          # Maximum search steps (default: 5)
--max_states         # Maximum number of states to maintain (default: 1)
--pruning_threshold  # Pruning threshold (default: 0.003)
```

#### Other Arguments

```bash
--selector       # Use feature selector: 1=yes, 0=no (default: 1)
--enlarge_num    # Feature expansion multiplier (default: 3)
--metadata_cat   # Metadata category (default: 3)
--max_tokens     # LLM max tokens (default: 8192)
--temperature    # LLM temperature (default: 1.0)
--log_path       # Log directory (default: ./log)
--log_filename   # Log filename (auto-generated)
```

### Usage Examples

```bash
# Basic CoT
python latte.py --method CoT --data_name credit-g --llm_model gpt-4o --iter 50

# CoT + Top-k history feedback
python latte.py --method CoT --data_name credit-g --history 1 --top 1

# Critic method
python latte.py --method Critic --data_name credit-g --iter 30

# OPRO dialogue optimization (5 dialogue turns x 10 iterations)
python latte.py --method OPRO --data_name credit-g --dialogue_turns 5 --iter 10

# OPROc (decision tree rule dialogue)
python latte.py --method OPROc --data_name credit-g --dialogue_turns 10

# ECoT evolution
python latte.py --method ECoT --data_name credit-g --ind_num 90 --iter 50

# Evo general evolution (supports multiple formats)
python latte.py --method Evo --data_name credit-g --output_format NL --iter 100

# ToT Tree of Thought
python latte.py --method ToT --data_name credit-g --num_thoughts 3 --max_steps 5
```

---

## bench.py - Benchmark Tool

`bench.py` is used for batch experiment execution, supporting combinations of multiple methods, datasets, and seeds.

### Features

- **Batch Execution**: Automatically combines methods, datasets, and seeds for experiments
- **Resume Support**: Detects existing log files and automatically skips completed experiments
- **Log-Based Metrics**: Parses log files to extract real metrics (time, tokens, val, test, AutoGluon test)
- **Success Rate**: Calculates per-iteration success rate from Error/Warning occurrences in logs
- **Configuration Persistence**: Supports saving/loading experiment configurations
- **Dry Run**: Preview mode that displays commands without executing them

### Command-Line Arguments

```bash
# Select what to run
--methods        # List of methods (default: all)
--datasets       # List of datasets (default: all)
--seeds          # List of seeds (default: 1 2 3 4 5 6)
--output_formats # List of output formats (optional, uses method-recommended formats by default)

# Model configuration
--llm_model      # LLM model (default: gpt-4o)
--task_type      # Task type (default: 1)
--metadata_cat   # Metadata category (default: 3)

# Iteration control
--iter           # Unified iteration count (optional, overrides method defaults)

# Method-specific arguments
--history        # CoT history feedback
--top            # CoT Top-k feedback
--dialogue_turns # OPRO/OPROc dialogue turns
--ind_num        # Evo/ECoT island individual count
...

# Execution control
--log_path       # Log directory (default: ./log)
--dry_run        # Preview mode, do not execute
--force          # Force re-run, ignore existing logs
--verbose        # Verbose output

# Configuration management
--save_config    # Save configuration to JSON file
--load_config    # Load configuration from JSON file
```

### Usage Examples

```bash
# Run all methods, all datasets, all seeds
python bench.py

# Run only specified methods
python bench.py --methods CoT Critic OPRO

# Run only specified datasets
python bench.py --datasets credit-g vehicle kc1

# Run only specified seeds
python bench.py --seeds 1 2 3

# Use a specific LLM model
python bench.py --llm_model gpt-4o-mini

# Preview mode (display commands without executing)
python bench.py --dry_run

# Force re-run
python bench.py --force

# Save configuration
python bench.py --save_config my_config.json --dry_run

# Load configuration and run
python bench.py --load_config my_config.json

# Verbose output
python bench.py --verbose --methods CoT --datasets credit-g --seeds 1
```

### Output Description

During execution, the following is displayed:

```
======================================================================
LATTEBench Benchmark Runner
======================================================================
Methods: ['CoT', 'Critic']
Datasets: ['credit-g', 'vehicle']
Seeds: [1, 2, 3]
LLM Model: gpt-4o
Total experiments: 12
======================================================================

[1/12] RUN: CoT | credit-g | NL | seed=1
         SUCCESS (45.2s)

[2/12] SKIP: CoT | credit-g | NL | seed=2
         Log exists: credit-g_CoT_gpt-4o_3_2.log
...
```

After execution, metrics are parsed from log files and summarized by method:

```
======================================================================
BENCHMARK SUMMARY
======================================================================
Total wall time: 1234.5s (20.6m)
Completed: 10
Skipped:   2
Failed:    0

----------------------------------------------------------------------
METRICS BY METHOD (parsed from log files)
----------------------------------------------------------------------

[CoT]
  Avg Time:         312.45s (6 runs)
  Avg Tokens:       48350 (6 runs)
  Avg Val (Best):   0.7789 (6 runs)
  Avg Test (RF):    0.7623 (6 runs)
  Avg Test (AG):    0.7856 (6 runs)
  Avg Success Rate: 82.00% (overall 246/300=82.00%, 6 runs)

[Critic]
  Avg Time:         425.12s (6 runs)
  Avg Tokens:       92100 (6 runs)
  Avg Val (Best):   0.7834 (6 runs)
  Avg Test (RF):    0.7701 (6 runs)
  Avg Test (AG):    0.7912 (6 runs)
  Avg Success Rate: 78.67% (overall 236/300=78.67%, 6 runs)

======================================================================

Results saved to: ./log/bench_results_20240101_120000.json
```

#### Metrics Description

All metrics are extracted by parsing each experiment's log file (produced by `latte.py`):

| Metric | Source in Log | Description |
|--------|--------------|-------------|
| Avg Time | `Total time used = ... seconds` | Average wall-clock time per experiment |
| Avg Tokens | `Total token usage = ...` | Average total LLM token consumption |
| Avg Val (Best) | `Best performance = ...` | Average best validation accuracy |
| Avg Test (RF) | `final_test_acc_rf = ...` | Average final test accuracy (RandomForest) |
| Avg Test (AG) | `final_test_acc_ag = ...` | Average final test accuracy (AutoGluon) |
| Avg Success Rate | Error/Warning per iteration | Ratio of iterations with no Error/Warning |

**Success Rate Details:**
- An iteration is counted as **successful** if it contains no `ERROR`, `WARNING` (log level), or `Warning:` (message content)
- For dialogue methods (OPRO, OPROc), an iteration is successful as long as **at least one dialogue turn** within it has no Error/Warning
- Experiments that did not complete (no `========== END ==========` in log) are excluded

### Default Method Configurations

| Method | Default Iterations | Recommended Output Formats |
|--------|-------------------|---------------------------|
| CoT | 50 | NL, cRPN, Code |
| Critic | 50 | NL, cRPN, Code |
| OPRO | 10 | NL, cRPN, Code |
| OPROc | 10 | Rule |
| ECoT | 50 | cRPN |
| Evo | 100 | NL, cRPN, Code |
| ToT | 1 | NL, cRPN |

---

## test_bench.py - Quick Functional Test

`test_bench.py` is a minimal test script for quickly verifying bench.py functionality without running full experiments.

### Features

- **Fast Testing**: Uses minimal iterations (2), single seed (1), single small dataset (balance-scale)
- **Quick Coverage**: Tests 2 methods (CoT + OPRO) covering both regular and dialogue-based methods
- **Isolated Logs**: Uses separate `./log_test` directory to avoid interfering with real experiments
- **Verification**: Automatically checks log file generation, parsing, and success rate calculation

### Command-Line Arguments

```bash
--dry_run      # Show commands without executing
--skip_run     # Skip running experiments, only test log parsing on existing logs
--methods      # Override test methods (default: CoT, OPRO)
--datasets     # Override test datasets (default: balance-scale)
--seeds        # Override test seeds (default: [1])
--iter         # Override iteration count (default: 2)
--keep_logs    # Keep test logs after completion (default: cleanup)
--verbose      # Verbose output
```

### Usage Examples

```bash
# Run full functional test (takes a few minutes with real LLM calls)
python test_bench.py

# Dry run to verify script works without LLM calls
python test_bench.py --dry_run

# Test only log parsing on existing test logs
python test_bench.py --skip_run

# Keep test logs for inspection
python test_bench.py --keep_logs

# Test with different methods
python test_bench.py --methods CoT Critic --iter 3
```

### Output Example

```
======================================================================
BENCH.PY FUNCTIONAL TEST
======================================================================
Test started at: 2024-01-15 10:00:00

Test Configuration:
  Methods:        ['CoT', 'OPRO']
  Datasets:       ['balance-scale']
  Seeds:          [1]
  Iterations:     2
  Dialogue turns: 2
  Log path:       ./log_test

  Total experiments: 2

======================================================================
Running bench.py with test configuration
======================================================================
Command: python bench.py --methods CoT OPRO --datasets balance-scale ...

[1/2] RUN: CoT | balance-scale | NL | seed=1
         SUCCESS (15.2s)
[2/2] RUN: OPRO | balance-scale | NL | seed=1
         SUCCESS (22.3s)

======================================================================
Verifying log files
======================================================================
  [OK] balance-scale_CoT_gpt-4o_3_1.log (8542 bytes)
  [OK] balance-scale_OPRO_gpt-4o_3_1.log (12341 bytes)

  Found 1 bench results file(s):
    - bench_results_20240115_100000.json (3456 bytes)

======================================================================
Testing log parsing functions
======================================================================

  Parsing: balance-scale_CoT_gpt-4o_3_1.log
    Metrics:
      total_time: 15.12
      total_tokens: 4532
      best_val: 0.7891
      final_test_rf: 0.7654
      final_test_ag: 0.7823
    Success Rate:
      total_iters: 2
      success_iters: 2
      success_rate: 100.00%

  Parsing: balance-scale_OPRO_gpt-4o_3_1.log
    Metrics:
      total_time: 22.05
      total_tokens: 8912
      best_val: 0.7934
      final_test_rf: 0.7701
      final_test_ag: 0.7889
    Success Rate:
      total_iters: 2
      success_iters: 1
      success_rate: 50.00%

======================================================================
TEST SUMMARY
======================================================================
All tests PASSED!

Cleaning up test logs in './log_test'...
Done.

Test finished at: 2024-01-15 10:02:30
```

---

## baseline_eval.py - Baseline Performance Evaluation

`baseline_eval.py` evaluates model performance on raw features (without feature engineering) as a comparison baseline for feature engineering methods.

### Features

- Train RandomForest and/or AutoGluon on raw features
- Support batch evaluation across multiple datasets and seeds
- Output validation and test set accuracy
- Generate table format for paper/report use

### Command-Line Arguments

```bash
# Dataset selection
--data_name      # Single dataset
--datasets       # Multiple datasets

# Seed selection
--seed           # Single seed
--seeds          # Multiple seeds

# Evaluation configuration
--downstream     # Downstream model: rf, ag, both (default: both)
--task_type      # Task type: 1=classification, 0=regression (default: 1)
--test_size      # Test set ratio (default: 0.2)
--val_size       # Validation set ratio (default: 0.2)

# Output
--log_path       # Log directory (default: ./log)
--output_json    # Result JSON file path
--verbose        # Verbose output
```

### Usage Examples

```bash
# Evaluate single dataset, single seed
python baseline_eval.py --data_name credit-g --seed 1

# Evaluate single dataset, multiple seeds
python baseline_eval.py --data_name credit-g --seeds 1 2 3 4 5 6

# Evaluate multiple datasets
python baseline_eval.py --datasets credit-g vehicle kc1 --seeds 1 2 3

# Use only RandomForest (faster)
python baseline_eval.py --data_name credit-g --downstream rf

# Use only AutoGluon
python baseline_eval.py --data_name credit-g --downstream ag

# Evaluate all datasets (using default parameters)
python baseline_eval.py
```

### Output Example

```
======================================================================
Baseline Performance Evaluation
======================================================================
Datasets: ['credit-g']
Seeds: [1, 2, 3]
Downstream: both
======================================================================

[1/3] Evaluating: credit-g | seed=1
    Features: 20, Samples: 600
    RF  - Val: 0.7500, Test: 0.7300
    AG  - Val: 0.7650, Test: 0.7450

======================================================================
TABLE FORMAT (for paper/report)
======================================================================
Dataset              RF Val     RF Test    AG Val     AG Test
----------------------------------------------------------------------
credit-g             0.7467     0.7267     0.7633     0.7417
======================================================================
```

---

## baseline_autofeat.py - AutoFeat Traditional Feature Engineering Baseline

`baseline_autofeat.py` uses the [AutoFeat](https://github.com/cod3licious/autofeat) library for traditional automated feature engineering, serving as a non-LLM comparison baseline.

### Features

- Use AutoFeat to automatically generate polynomial and interaction features
- Automatic handling of missing values and categorical variable encoding
- Evaluate RandomForest and AutoGluon performance

### Dependencies

```bash
pip install autofeat
```

### Command-Line Arguments

```bash
--data_name       # Dataset name (default: credit-g)
--seed            # Random seed (default: 2)
--test_size       # Test set ratio (default: 0.2)
--val_size        # Validation set ratio (default: 0.2)

# AutoFeat arguments
--feateng_steps   # Feature engineering steps (default: 2)
--max_gb          # Maximum memory usage in GB (default: 1)
--n_jobs          # Number of parallel jobs (default: 10)

--log_path        # Log directory (default: ./log)
--log_filename    # Log filename
```

### Usage Examples

```bash
# Basic usage
python baseline_autofeat.py --data_name credit-g --seed 1

# Increase feature engineering steps (generates more features, but slower)
python baseline_autofeat.py --data_name credit-g --feateng_steps 3

# Multiple seeds require manual looping
for seed in 1 2 3 4 5 6; do
    python baseline_autofeat.py --data_name credit-g --seed $seed
done
```

### Output

- Log files: `./log/{data_name}_AutoFeat_{seed}.log`
- Generated feature data: `./tmp/{data_name}/train.csv`, `val.csv`, `test.csv`

---

## Datasets

The framework supports the following datasets (located in the `data/` directory):

| Dataset | Description |
|---------|-------------|
| credit-g | German Credit Assessment |
| credit-approval | Credit Approval |
| kc1 | KC1 Software Defect Prediction |
| qsar-biodeg | QSAR Biodegradability Prediction |
| vehicle | Vehicle Classification |
| heart-h | Heart Disease Prediction |
| electricity | Electricity Demand Prediction |
| balance-scale | Balance Scale Classification |

---

## Logs and Results

- **Log files**: `./log/{data_name}_{method}_{llm_model}_{metadata_cat}_{seed}.log`
- **Best model**: `./tmp/{data_name}/best_train.csv`, `best_test.csv`
- **Metadata**: `./tmp/{data_name}/metadata.json`
- **Benchmark results**: `./log/bench_results_{timestamp}.json`
- **Baseline performance results**: `./log/baseline_results_{timestamp}.json`

---

## Typical Workflow

```bash
# 0a. Evaluate baseline performance (raw features, no feature engineering)
python baseline_eval.py --datasets credit-g vehicle --seeds 1 2 3

# 0b. Evaluate traditional feature engineering baseline (AutoFeat)
for seed in 1 2 3; do
    python baseline_autofeat.py --data_name credit-g --seed $seed
done

# 1. Preview experiment plan
python bench.py --methods CoT OPRO --datasets credit-g vehicle --seeds 1 2 3 --dry_run

# 2. Save configuration
python bench.py --methods CoT OPRO --datasets credit-g vehicle --seeds 1 2 3 \
    --save_config experiment_config.json --dry_run

# 3. Execute experiments
python bench.py --load_config experiment_config.json

# 4. Monitor logs
tail -f ./log/credit-g_CoT_gpt-4o_3_1.log

# 5. Analyze results
cat ./log/bench_results_*.json | python -m json.tool

# 6. Compare performance
# Raw baseline:      ./log/baseline_results_*.json
# AutoFeat baseline: ./log/{data_name}_AutoFeat_{seed}.log
# LLM method results: ./log/bench_results_*.json and final_test_acc in each method's log
```
