# Multi-Armed Bandit Simulation Dashboard

This is a web-based dashboard for simulating and visualizing Thompson Sampling Multi-Armed Bandit (TS-MAB) experiments for email subject line optimization.

## Features

- Configure simulation parameters:
  - Number of weeks to simulate (2-20)
  - Number of initial subject lines (5-10)
  - Maximum number of subject lines (5-20)
  - Epsilon allocation for new subject lines (0.1-0.5)
  - New subject line allocation period (1-5 weeks)
  - **Customize open rates for each subject line** (optional)
- Dynamic subject line allocation:
  - When a subject line reaches zero distribution, a new one is introduced
  - New subject lines receive an epsilon allocation for a configurable number of weeks
- Multiple visualization tabs:
  - Simulation log showing detailed progress
  - Summary of key metrics and final results
  - Metrics tab for analyzing exploration vs exploitation trade-offs
  - Interactive visualizations of distribution, CTR, and performance
  - Raw data explorer

## Requirements

- Python 3.7+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd mab_sims_queue
```

2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Set simulation parameters using the sidebar controls:
   - Number of weeks to simulate
   - Number of initial subject lines
   - Maximum subject lines in the pool
   - Epsilon allocation percentage for new subject lines
   - Duration of epsilon allocation for new subject lines
   - **Open rate customization (optional)**: Enable "Customize Open Rates" to set min/max open rates for each subject line
2. Click "Run Simulation" to start
3. Explore the different tabs to analyze the results

## How the Simulation Works

The simulation models email campaign optimization over several weeks with these key behaviors:

1. **Initial Phase**: Start with a specified number of subject lines, each with equal distribution
2. **Subject Line Dynamics**:
   - When a subject line drops to 0% distribution, a new one is introduced from the pool
   - New subject lines receive a guaranteed epsilon allocation for a specified number of weeks
   - The remaining (1-epsilon) allocation is distributed among standard subject lines using Thompson Sampling

3. **Tracking and Optimization**:
   - Each subject line has a random open rate range assigned at the start
   - The simulation tracks cumulative performance metrics for better long-term optimization
   - Subject lines can have three statuses: inactive, new, and standard

4. **Simulation Completion**:
   - The simulation runs until the maximum number of weeks is reached
   - Or it ends early if any subject line reaches 100% distribution

### Open Rate Customization

The dashboard allows you to customize the open rate ranges for each subject line:

1. **Default Mode**: If you don't customize open rates, the system assigns random open rate ranges:
   - Minimum rates between 10-25%
   - Maximum rates 5-15 percentage points higher than the minimum

2. **Custom Mode**: Enable "Customize Open Rates" to precisely control each subject line's performance:
   - Set min/max open rates for each subject line individually
   - Initial subject lines and additional subject lines are organized in separate expandable sections
   - The system enforces that max rates are always higher than min rates

This feature allows you to test specific scenarios, such as:
- A scenario with one clearly superior subject line
- Subject lines with similar performance
- New subject lines that perform better than initial ones
- Progressive improvements in subject line quality

## Files

- `app.py`: The Streamlit web application
- `simulation.py`: Contains the simulation logic for the TS-MAB experiments
- `ts_mab.py`: Implementation of the Thompson Sampling Multi-Armed Bandit algorithm
- `run.sh`: Shell script to easily run the application 