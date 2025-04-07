import pandas as pd
import numpy as np
from ts_mab import run_ts_mab

def generate_weekly_sends(n_weeks=10, n_subjectlines=5, max_subjectlines=15, epsilon=0.2, new_sl_weeks=2, custom_min_rates=None, custom_max_rates=None, verbose=True):
    """
    Generate random total sends for each week.
    
    Parameters:
    -----------
    n_weeks : int, default=10
        Number of weeks to simulate
    n_subjectlines : int, default=5
        Initial number of subject lines
    max_subjectlines : int, default=15
        Maximum number of subject lines in the pool
    epsilon : float, default=0.2
        Percentage of allocation for new subject lines
    new_sl_weeks : int, default=2
        Number of weeks a new subject line will get the epsilon allocation
    custom_min_rates : dict, default=None
        Dictionary mapping subject line names to minimum open rates
    custom_max_rates : dict, default=None
        Dictionary mapping subject line names to maximum open rates
    verbose : bool, default=True
        Whether to print detailed simulation information
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with complete simulation results
    """
    # Validate input parameters
    validate_inputs(n_subjectlines, max_subjectlines)
    
    # Initialize simulation data
    simulation_data = initialize_simulation(n_subjectlines, max_subjectlines, custom_min_rates, custom_max_rates, verbose)
    
    # Run weekly simulation
    run_weekly_simulation(simulation_data, n_weeks, epsilon, new_sl_weeks, verbose)
    
    # Create comprehensive results DataFrame
    results_df = create_results_dataframe(simulation_data, n_weeks)
    
    return results_df


def validate_inputs(n_subjectlines, max_subjectlines):
    """Validate input parameters for the simulation."""
    if n_subjectlines > max_subjectlines:
        raise ValueError("Initial subject lines cannot exceed the maximum subject lines")
    
    if n_subjectlines < 5:
        n_subjectlines = 5
        print("Warning: Minimum number of subject lines is 5. Using 5 initial subject lines.")
    
    return n_subjectlines


def initialize_simulation(n_subjectlines, max_subjectlines, custom_min_rates, custom_max_rates, verbose=True):
    """Initialize all data structures needed for the simulation."""
    # Create subject line lists
    all_sls = [f"SL_{i}" for i in range(1, max_subjectlines + 1)]
    active_sls = all_sls[:n_subjectlines]
    available_sls = all_sls[n_subjectlines:]
    
    # Track which SLs are currently active and their introduction weeks
    active_sl_indices = list(range(n_subjectlines))
    sl_introduction_weeks = {i: 1 for i in range(n_subjectlines)}  # SL index -> week introduced
    sl_status = {i: "standard" for i in range(n_subjectlines)}  # SL index -> status ("new" or "standard")
    
    # Track cumulative sends and opens for all possible subject lines
    cumulative_sends = [0] * max_subjectlines
    cumulative_opens = [0] * max_subjectlines
    
    # Track which SLs have zero distribution
    zero_dist_sls = set()
    new_zero_dist_sls = set()
    
    # Generate open rate ranges for all possible subject lines
    sl_open_rate_ranges = generate_open_rate_ranges(all_sls, custom_min_rates, custom_max_rates, verbose)
    
    # Create lists to store all simulation data
    all_weeks = []
    all_total_sends = []
    all_sl_distributions = []  # This will store distributions for all subject lines (active and inactive)
    all_sl_sends = []  # This will store sends for all subject lines (active and inactive)
    all_sl_opens = []  # This will store opens for all subject lines (active and inactive)
    all_cumulative_sends = []
    all_cumulative_opens = []
    all_active_sls = []  # Track which SLs are active in each week
    all_sl_statuses = []  # Track status of each SL in each week
    
    # Bundle all data in a dictionary
    simulation_data = {
        'n_subjectlines': n_subjectlines,
        'max_subjectlines': max_subjectlines,
        'all_sls': all_sls,
        'active_sls': active_sls,
        'available_sls': available_sls,
        'active_sl_indices': active_sl_indices,
        'sl_introduction_weeks': sl_introduction_weeks,
        'sl_status': sl_status,
        'cumulative_sends': cumulative_sends,
        'cumulative_opens': cumulative_opens,
        'zero_dist_sls': zero_dist_sls,
        'new_zero_dist_sls': new_zero_dist_sls,
        'sl_open_rate_ranges': sl_open_rate_ranges,
        'all_weeks': all_weeks,
        'all_total_sends': all_total_sends,
        'all_sl_distributions': all_sl_distributions,
        'all_sl_sends': all_sl_sends,
        'all_sl_opens': all_sl_opens,
        'all_cumulative_sends': all_cumulative_sends,
        'all_cumulative_opens': all_cumulative_opens,
        'all_active_sls': all_active_sls,
        'all_sl_statuses': all_sl_statuses,
        'next_dist': None  # Will be set during simulation
    }
    
    return simulation_data


def generate_open_rate_ranges(all_sls, custom_min_rates, custom_max_rates, verbose=True):
    """Generate open rate ranges for all subject lines."""
    sl_open_rate_ranges = {}
    for sl in all_sls:
        if custom_min_rates and custom_max_rates and sl in custom_min_rates and sl in custom_max_rates:
            # Use custom rates if provided
            min_rate = custom_min_rates[sl]
            max_rate = custom_max_rates[sl]
        else:
            # Otherwise generate random rates
            min_rate = np.random.uniform(0.1, 0.25)
            max_rate = min_rate + np.random.uniform(0.05, 0.15)
        
        sl_open_rate_ranges[sl] = (min_rate, max_rate)
    
    # Print the open rate ranges being used
    if verbose:
        print("Subject Line Open Rate Ranges:")
        for sl in all_sls:
            min_rate, max_rate = sl_open_rate_ranges[sl]
            print(f"{sl}: {min_rate*100:.2f}% - {max_rate*100:.2f}%")
        print("\n")
    
    return sl_open_rate_ranges


def run_weekly_simulation(data, n_weeks, epsilon, new_sl_weeks, verbose=True):
    """Run the MAB simulation week by week."""
    for week in range(1, n_weeks + 1):
        if verbose:
            print(f"Week {week}")
        
        # Generate random total sends for this week
        week_total_sends = np.random.randint(2000, 5001)
        
        # Update subject line statuses and track active SLs
        update_subject_line_status(data, week, new_sl_weeks, verbose)
        
        # Process the current week's data
        if week == 1:
            process_first_week(data, week_total_sends, verbose)
        else:
            process_subsequent_week(data, week, week_total_sends, epsilon, new_sl_weeks, verbose)
        
        # Check if simulation should end (a SL reached 100% distribution)
        if check_for_simulation_end(data, verbose):
            break


def update_subject_line_status(data, week, new_sl_weeks, verbose=True):
    """Update status of subject lines based on current week."""
    # Track active SLs for this week
    current_active_sls = [i for i in data['active_sl_indices'] if i not in data['zero_dist_sls']]
    data['all_active_sls'].append(current_active_sls.copy())
    
    # Update status of subject lines
    for sl_idx in data['active_sl_indices']:
        if data['sl_status'][sl_idx] == "new" and week - data['sl_introduction_weeks'][sl_idx] >= new_sl_weeks:
            data['sl_status'][sl_idx] = "standard"
    
    # Current status of all subject lines
    current_sl_statuses = ["inactive"] * data['max_subjectlines']
    for sl_idx in data['active_sl_indices']:
        current_sl_statuses[sl_idx] = data['sl_status'][sl_idx]
    data['all_sl_statuses'].append(current_sl_statuses.copy())
    
    if verbose:
        print(f"Active subject lines: {[data['all_sls'][i] for i in current_active_sls]}")


def process_first_week(data, week_total_sends, verbose=True):
    """Process the first week of the simulation."""
    # First week has uniform distribution among initial subject lines
    n_subjectlines = data['n_subjectlines']
    sends_per_sl = week_total_sends // n_subjectlines
    
    # Initialize sends and opens arrays for all possible subject lines
    sends_arr = [0] * data['max_subjectlines']
    opens_arr = [0] * data['max_subjectlines']
    
    # Process sends and opens for each initial subject line
    for idx in range(n_subjectlines):
        sends_arr[idx] = sends_per_sl
        opens_arr[idx] = calculate_opens(data, idx, sends_per_sl)
    
    # Update cumulative tracking
    update_cumulative_metrics(data, sends_arr, opens_arr)
    
    if verbose:
        print(f"Sends this week: {sends_arr[:n_subjectlines]}")
        print(f"Opens this week: {opens_arr[:n_subjectlines]}")
        print(f"Cumulative sends: {data['cumulative_sends'][:n_subjectlines]}")
        print(f"Cumulative opens: {data['cumulative_opens'][:n_subjectlines]}")
    
    # Run MAB on initial subject lines
    initial_sends = [data['cumulative_sends'][i] for i in range(n_subjectlines)]
    initial_opens = [data['cumulative_opens'][i] for i in range(n_subjectlines)]
    
    next_dist_filtered = run_ts_mab(initial_sends, initial_opens)
    
    # Create full distribution array for all possible subject lines
    next_dist = [0] * data['max_subjectlines']
    for i in range(n_subjectlines):
        next_dist[i] = next_dist_filtered[i]
    
    data['next_dist'] = next_dist
    
    # Check if any SL has 100% distribution
    if 100 in next_dist and verbose:
        winner_idx = next_dist.index(100)
        print(f"Winner found! {data['all_sls'][winner_idx]} has 100% distribution.")
        print(f"Simulation ending early at week 1.")
    
    if verbose:
        print(f"Next distribution: {next_dist[:n_subjectlines]}")
    
    # Store data for this week
    distribution = [0] * data['max_subjectlines']
    for i in range(n_subjectlines):
        distribution[i] = 1.0/n_subjectlines  # First week is uniform
    
    store_weekly_data(data, 1, week_total_sends, distribution, sends_arr, opens_arr)
    
    # If any distribution is 0, prepare to add new subject lines next week
    new_zero_dist_sls = set([i for i in range(n_subjectlines) if next_dist[i] == 0])
    data['zero_dist_sls'].update(new_zero_dist_sls)
    data['new_zero_dist_sls'] = new_zero_dist_sls
    
    if new_zero_dist_sls and data['available_sls'] and verbose:
        print(f"Subject lines with zero distribution: {[data['all_sls'][i] for i in new_zero_dist_sls]}")
        print(f"Will add new subject lines next week.")


def process_subsequent_week(data, week, week_total_sends, epsilon, new_sl_weeks, verbose=True):
    """Process weeks after the first week."""
    current_dist = data['next_dist']
    if verbose:
        print(f"Current distribution: {[current_dist[i] for i in data['active_sl_indices']]}")
    
    # Check if new subject lines need to be added
    num_new_zero_dist = len(data['new_zero_dist_sls'])
    new_sls_to_add = min(num_new_zero_dist, len(data['available_sls']))
    
    # Reset new_zero_dist_sls for this week (will be populated at the end)
    data['new_zero_dist_sls'] = set()
    
    # Add new subject lines if needed
    if new_sls_to_add > 0 and data['available_sls']:
        add_new_subject_lines(data, new_sls_to_add, week, verbose)
    
    # Calculate allocations for this week
    standard_sls, new_sls, epsilon_allocation, standard_total_allocation = calculate_allocations(
        data, epsilon)
    
    # Calculate distribution for each subject line
    calculate_distribution(data, standard_sls, new_sls, epsilon_allocation, standard_total_allocation)
    
    # Distribute sends and opens based on the current distribution
    sends_arr, opens_arr = distribute_sends_and_opens(data, week_total_sends, current_dist)
    
    # Update cumulative metrics
    update_cumulative_metrics(data, sends_arr, opens_arr)
    
    if verbose:
        print(f"Sends this week: {[sends_arr[i] for i in data['active_sl_indices']]}")
        print(f"Opens this week: {[opens_arr[i] for i in data['active_sl_indices']]}")
        print(f"Cumulative sends: {[data['cumulative_sends'][i] for i in data['active_sl_indices']]}")
        print(f"Cumulative opens: {[data['cumulative_opens'][i] for i in data['active_sl_indices']]}")
    
    # Calculate next week's distribution
    calculate_next_distribution(data, verbose)
    
    # Store weekly data
    store_weekly_data(data, week, week_total_sends, current_dist, sends_arr, opens_arr)
    
    # Identify new zero distribution SLs for the next week
    identify_new_zero_distribution_sls(data, verbose)


def calculate_opens(data, sl_idx, sends):
    """Calculate opens based on open rate range for a subject line."""
    if sends == 0:
        return 0
    
    sl_name = data['all_sls'][sl_idx]
    min_rate, max_rate = data['sl_open_rate_ranges'][sl_name]
    return round(np.random.uniform(min_rate, max_rate) * sends)


def update_cumulative_metrics(data, sends_arr, opens_arr):
    """Update cumulative sends and opens for all subject lines."""
    for i in range(data['max_subjectlines']):
        data['cumulative_sends'][i] += sends_arr[i]
        data['cumulative_opens'][i] += opens_arr[i]


def store_weekly_data(data, week, week_total_sends, distribution, sends_arr, opens_arr):
    """Store weekly simulation data."""
    data['all_weeks'].append(week)
    data['all_total_sends'].append(week_total_sends)
    data['all_sl_distributions'].append(distribution.copy())
    data['all_sl_sends'].append(sends_arr)
    data['all_sl_opens'].append(opens_arr)
    data['all_cumulative_sends'].append(data['cumulative_sends'].copy())
    data['all_cumulative_opens'].append(data['cumulative_opens'].copy())


def add_new_subject_lines(data, new_sls_to_add, week, verbose=True):
    """Add new subject lines from the available pool."""
    new_sl_indices = []
    for _ in range(new_sls_to_add):
        if data['available_sls']:
            new_sl = data['available_sls'].pop(0)
            new_sl_idx = data['all_sls'].index(new_sl)
            data['active_sl_indices'].append(new_sl_idx)
            new_sl_indices.append(new_sl_idx)
            data['sl_introduction_weeks'][new_sl_idx] = week
            data['sl_status'][new_sl_idx] = "new"
    
    if verbose and new_sl_indices:
        print(f"Added new subject lines: {[data['all_sls'][i] for i in new_sl_indices]}")
    
    # Update current active SLs
    current_active_sls = [i for i in data['active_sl_indices'] if i not in data['zero_dist_sls']]
    data['all_active_sls'][-1] = current_active_sls.copy()  # Update the latest entry
    
    # Update statuses
    for sl_idx in data['active_sl_indices']:
        data['all_sl_statuses'][-1][sl_idx] = data['sl_status'][sl_idx]  # Update the latest entry


def calculate_allocations(data, epsilon):
    """Calculate allocation percentages for new and standard subject lines."""
    # Count how many "new" subject lines we have
    new_sls = [idx for idx in data['active_sl_indices'] 
             if data['sl_status'][idx] == "new"]
    standard_sls = [idx for idx in data['active_sl_indices'] 
                  if data['sl_status'][idx] == "standard" and idx not in data['zero_dist_sls']]
    
    # Calculate epsilon allocation if we have new subject lines
    epsilon_allocation = {}
    standard_total_allocation = 1.0
    
    if new_sls:
        # Distribute epsilon evenly among new subject lines
        epsilon_per_sl = epsilon / len(new_sls)
        for sl_idx in new_sls:
            epsilon_allocation[sl_idx] = epsilon_per_sl
        
        # Remaining allocation for standard subject lines
        standard_total_allocation = 1.0 - epsilon
    
    return standard_sls, new_sls, epsilon_allocation, standard_total_allocation


def calculate_distribution(data, standard_sls, new_sls, epsilon_allocation, standard_total_allocation):
    """Calculate distribution for each subject line."""
    current_dist = data['next_dist']
    
    # For standard subject lines, use MAB allocation
    if standard_sls:
        # Run MAB on standard subject lines using cumulative data
        standard_sends = [data['cumulative_sends'][i] for i in standard_sls]
        standard_opens = [data['cumulative_opens'][i] for i in standard_sls]
        
        standard_dist = run_ts_mab(standard_sends, standard_opens)
        
        # Scale the standard distribution to use only standard_total_allocation
        for i, sl_idx in enumerate(standard_sls):
            current_dist[sl_idx] = standard_dist[i] * standard_total_allocation / 100
    
    # Add epsilon allocation for new subject lines
    for sl_idx, alloc in epsilon_allocation.items():
        current_dist[sl_idx] = alloc
    
    # Ensure zero distribution for SLs that are supposed to have zero
    for sl_idx in data['zero_dist_sls']:
        current_dist[sl_idx] = 0


def distribute_sends_and_opens(data, week_total_sends, current_dist):
    """Distribute sends and opens based on the current distribution."""
    sends_arr = [0] * data['max_subjectlines']
    opens_arr = [0] * data['max_subjectlines']
    
    for i in range(data['max_subjectlines']):
        if i in data['active_sl_indices'] and i not in data['zero_dist_sls']:
            sends_arr[i] = round(current_dist[i] * week_total_sends)
            opens_arr[i] = calculate_opens(data, i, sends_arr[i])
    
    return sends_arr, opens_arr


def calculate_next_distribution(data, verbose=True):
    """Calculate distribution for the next week."""
    # Get non-zero distribution active subject lines
    active_non_zero_sls = [i for i in data['active_sl_indices'] if i not in data['zero_dist_sls']]
    
    if active_non_zero_sls:
        active_sends = [data['cumulative_sends'][i] for i in active_non_zero_sls]
        active_opens = [data['cumulative_opens'][i] for i in active_non_zero_sls]
        
        next_dist_filtered = run_ts_mab(active_sends, active_opens)
        
        # Create full distribution array
        next_dist = [0] * data['max_subjectlines']
        for i, sl_idx in enumerate(active_non_zero_sls):
            next_dist[sl_idx] = next_dist_filtered[i]
        
        data['next_dist'] = next_dist
        
        if verbose:
            print(f"Next distribution: {[next_dist[i] for i in data['active_sl_indices']]}")
    else:
        # No active subject lines with non-zero distribution
        data['next_dist'] = [0] * data['max_subjectlines']
        if verbose:
            print("No active subject lines with non-zero distribution.")


def identify_new_zero_distribution_sls(data, verbose=True):
    """Identify subject lines that will have zero distribution next week."""
    # Only consider active SLs that had non-zero distribution this week but will have zero next week
    data['new_zero_dist_sls'] = set([i for i in data['active_sl_indices'] 
                               if i not in data['zero_dist_sls'] and data['next_dist'][i] == 0])
    
    # Update overall zero_dist_sls
    data['zero_dist_sls'].update(data['new_zero_dist_sls'])
    
    if verbose and data['new_zero_dist_sls']:
        print(f"Subject lines with zero distribution for next week: {[data['all_sls'][i] for i in data['new_zero_dist_sls']]}")


def check_for_simulation_end(data, verbose=True):
    """Check if any subject line has 100% distribution (end condition)."""
    if 100 in data['next_dist']:
        if verbose:
            winner_idx = data['next_dist'].index(100)
            print(f"Winner found! {data['all_sls'][winner_idx]} has 100% distribution.")
            print(f"Simulation ending early at week {data['all_weeks'][-1]}.")
        return True
    return False


def create_results_dataframe(data, n_weeks):
    """Create a comprehensive DataFrame with all simulation results."""
    results = []
    
    for i in range(len(data['all_weeks'])):
        week_data = {
            'week': data['all_weeks'][i],
            'total_sends': data['all_total_sends'][i],
            'active_sls': ','.join([data['all_sls'][idx] for idx in data['all_active_sls'][i]])
        }
        
        for j in range(data['max_subjectlines']):
            sl_name = data['all_sls'][j]
            week_data[f'{sl_name}_status'] = data['all_sl_statuses'][i][j]
            week_data[f'{sl_name}_distribution'] = data['all_sl_distributions'][i][j]
            week_data[f'{sl_name}_sends'] = data['all_sl_sends'][i][j]
            week_data[f'{sl_name}_opens'] = data['all_sl_opens'][i][j]
            week_data[f'{sl_name}_cumulative_sends'] = data['all_cumulative_sends'][i][j]
            week_data[f'{sl_name}_cumulative_opens'] = data['all_cumulative_opens'][i][j]
            
            # Calculate CTR
            if data['all_sl_sends'][i][j] > 0:
                week_data[f'{sl_name}_weekly_ctr'] = data['all_sl_opens'][i][j] / data['all_sl_sends'][i][j] * 100
            else:
                week_data[f'{sl_name}_weekly_ctr'] = 0
                
            if data['all_cumulative_sends'][i][j] > 0:
                week_data[f'{sl_name}_cumulative_ctr'] = data['all_cumulative_opens'][i][j] / data['all_cumulative_sends'][i][j] * 100
            else:
                week_data[f'{sl_name}_cumulative_ctr'] = 0
        
        results.append(week_data)
    
    # Add next week's predicted distribution if simulation didn't end with a winner
    if len(data['all_weeks']) == n_weeks:
        add_prediction_row(data, results, n_weeks)
    
    return pd.DataFrame(results)


def add_prediction_row(data, results, n_weeks):
    """Add a row for next week's predicted distribution."""
    next_week_data = {'week': n_weeks + 1, 'total_sends': 0}
    
    # Determine active SLs for the prediction
    active_sls_for_prediction = [i for i in data['active_sl_indices'] if i not in data['zero_dist_sls']]
    next_week_data['active_sls'] = ','.join([data['all_sls'][idx] for idx in active_sls_for_prediction])
    
    for j in range(data['max_subjectlines']):
        sl_name = data['all_sls'][j]
        
        if j in data['active_sl_indices']:
            # Carry forward the status from the last week
            next_week_data[f'{sl_name}_status'] = data['all_sl_statuses'][-1][j]
            
            # Prediction only makes sense for active SLs
            if j in active_sls_for_prediction:
                next_week_data[f'{sl_name}_distribution'] = data['next_dist'][j]
            else:
                next_week_data[f'{sl_name}_distribution'] = 0
        else:
            next_week_data[f'{sl_name}_status'] = "inactive"
            next_week_data[f'{sl_name}_distribution'] = 0
            
        next_week_data[f'{sl_name}_sends'] = 0
        next_week_data[f'{sl_name}_opens'] = 0
        next_week_data[f'{sl_name}_cumulative_sends'] = data['cumulative_sends'][j]
        next_week_data[f'{sl_name}_cumulative_opens'] = data['cumulative_opens'][j]
        next_week_data[f'{sl_name}_weekly_ctr'] = 0
        
        if data['cumulative_sends'][j] > 0:
            next_week_data[f'{sl_name}_cumulative_ctr'] = data['cumulative_opens'][j] / data['cumulative_sends'][j] * 100
        else:
            next_week_data[f'{sl_name}_cumulative_ctr'] = 0
    
    results.append(next_week_data)


if __name__ == "__main__":
    result_df = generate_weekly_sends(n_weeks=5, n_subjectlines=5, max_subjectlines=10, epsilon=0.2, new_sl_weeks=2)
    print("\nSimulation Results DataFrame:")
    print(result_df)
