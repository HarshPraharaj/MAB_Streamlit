import pandas as pd
import numpy as np
from ts_mab import run_ts_mab

def generate_weekly_sends(n_weeks=10, n_subjectlines=5, max_subjectlines=15, epsilon=0.2, new_sl_weeks=2, custom_min_rates=None, custom_max_rates=None):
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
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with complete simulation results
    """
   
    # Validate input parameters
    if n_subjectlines > max_subjectlines:
        raise ValueError("Initial subject lines cannot exceed the maximum subject lines")
    
    if n_subjectlines < 5:
        n_subjectlines = 5
        print("Warning: Minimum number of subject lines is 5. Using 5 initial subject lines.")

    # All possible subject lines in the pool
    all_sls = [f"SL_{i}" for i in range(1, max_subjectlines + 1)]
    
    # Initial active subject lines
    active_sls = all_sls[:n_subjectlines]
    
    # Available subject lines to introduce later
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
    
    # Track newly added zero distribution SLs for the current week
    new_zero_dist_sls = set()
    
    # Generate open rate ranges for all possible subject lines
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
    print("Subject Line Open Rate Ranges:")
    for sl in all_sls:
        min_rate, max_rate = sl_open_rate_ranges[sl]
        print(f"{sl}: {min_rate*100:.2f}% - {max_rate*100:.2f}%")
    print("\n")
    
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

    for week in range(1, n_weeks + 1):
        week_total_sends = np.random.randint(2000, 5001)
        
        # Track active SLs for this week
        current_active_sls = [i for i in active_sl_indices if i not in zero_dist_sls]
        all_active_sls.append(current_active_sls.copy())
        
        # Update status of subject lines
        for sl_idx in active_sl_indices:
            if sl_status[sl_idx] == "new" and week - sl_introduction_weeks[sl_idx] >= new_sl_weeks:
                sl_status[sl_idx] = "standard"
        
        # Current status of all subject lines
        current_sl_statuses = ["inactive"] * max_subjectlines
        for sl_idx in active_sl_indices:
            current_sl_statuses[sl_idx] = sl_status[sl_idx]
        all_sl_statuses.append(current_sl_statuses.copy())
        
        print(f"Week {week}")
        print(f"Active subject lines: {[all_sls[i] for i in current_active_sls]}")
        
        # Initialize sends and opens arrays for all possible subject lines
        sends_arr = [0] * max_subjectlines
        opens_arr = [0] * max_subjectlines
        
        if week == 1:
            # First week has uniform distribution among initial subject lines
            sends_per_sl = week_total_sends // n_subjectlines
            
            for idx in range(n_subjectlines):
                sends_arr[idx] = sends_per_sl
                opens_arr[idx] = round(np.random.uniform(
                    sl_open_rate_ranges[all_sls[idx]][0], 
                    sl_open_rate_ranges[all_sls[idx]][1]
                ) * sends_per_sl)
            
            # Update cumulative tracking
            for i in range(max_subjectlines):
                cumulative_sends[i] += sends_arr[i]
                cumulative_opens[i] += opens_arr[i]
                
            print(f"Sends this week: {sends_arr[:n_subjectlines]}")
            print(f"Opens this week: {opens_arr[:n_subjectlines]}")
            print(f"Cumulative sends: {cumulative_sends[:n_subjectlines]}")
            print(f"Cumulative opens: {cumulative_opens[:n_subjectlines]}")
            
            # For first week, run MAB on initial subject lines
            initial_sends = [cumulative_sends[i] for i in range(n_subjectlines)]
            initial_opens = [cumulative_opens[i] for i in range(n_subjectlines)]
            
            next_dist_filtered = run_ts_mab(initial_sends, initial_opens)
            
            # Create full distribution array for all possible subject lines
            next_dist = [0] * max_subjectlines
            for i in range(n_subjectlines):
                next_dist[i] = next_dist_filtered[i]
            
            # Check if any SL has 100% distribution
            if 100 in next_dist:
                winner_idx = next_dist.index(100)
                print(f"Winner found! {all_sls[winner_idx]} has 100% distribution.")
                print(f"Simulation ending early at week {week}.")
                # Will exit loop after this iteration
            
            print(f"Next distribution: {next_dist[:n_subjectlines]}")
            
            # Store data for this week
            distribution = [0] * max_subjectlines
            for i in range(n_subjectlines):
                distribution[i] = 1.0/n_subjectlines  # First week is uniform
            
            all_weeks.append(week)
            all_total_sends.append(week_total_sends)
            all_sl_distributions.append(distribution)
            all_sl_sends.append(sends_arr)
            all_sl_opens.append(opens_arr)
            all_cumulative_sends.append(cumulative_sends.copy())
            all_cumulative_opens.append(cumulative_opens.copy())
            
            # If any distribution is 0, prepare to add new subject lines next week
            new_zero_dist_sls = set([i for i in range(n_subjectlines) if next_dist[i] == 0])
            zero_dist_sls.update(new_zero_dist_sls)
            
            if new_zero_dist_sls and available_sls:
                print(f"Subject lines with zero distribution: {[all_sls[i] for i in new_zero_dist_sls]}")
                print(f"Will add new subject lines next week.")
        else:
            current_dist = next_dist
            print(f"Current distribution: {[current_dist[i] for i in active_sl_indices]}")
            
            # Check if new subject lines need to be added based on newly zero-distribution SLs from the previous week
            num_new_zero_dist = len(new_zero_dist_sls)
            new_sls_to_add = min(num_new_zero_dist, len(available_sls))
            
            # Reset new_zero_dist_sls for this week (will be populated at the end)
            new_zero_dist_sls = set()
            
            if new_sls_to_add > 0 and available_sls:
                # Add new subject lines from the available pool
                new_sl_indices = []
                for _ in range(new_sls_to_add):
                    if available_sls:
                        new_sl = available_sls.pop(0)
                        new_sl_idx = all_sls.index(new_sl)
                        active_sl_indices.append(new_sl_idx)
                        new_sl_indices.append(new_sl_idx)
                        sl_introduction_weeks[new_sl_idx] = week
                        sl_status[new_sl_idx] = "new"
                
                print(f"Added new subject lines: {[all_sls[i] for i in new_sl_indices]}")
                
                # Update current active SLs
                current_active_sls = [i for i in active_sl_indices if i not in zero_dist_sls]
                all_active_sls[-1] = current_active_sls.copy()  # Update the latest entry
                
                # Update statuses
                for sl_idx in active_sl_indices:
                    current_sl_statuses[sl_idx] = sl_status[sl_idx]
                all_sl_statuses[-1] = current_sl_statuses.copy()  # Update the latest entry
            
            # Count how many "new" subject lines we have
            new_sls = [idx for idx in active_sl_indices if sl_status[idx] == "new"]
            standard_sls = [idx for idx in active_sl_indices if sl_status[idx] == "standard" and idx not in zero_dist_sls]
            
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
            
            # For standard subject lines, use MAB allocation
            if standard_sls:
                # Run MAB on standard subject lines using cumulative data
                standard_sends = [cumulative_sends[i] for i in standard_sls]
                standard_opens = [cumulative_opens[i] for i in standard_sls]
                
                standard_dist = run_ts_mab(standard_sends, standard_opens)
                
                # Scale the standard distribution to use only standard_total_allocation
                for i, sl_idx in enumerate(standard_sls):
                    current_dist[sl_idx] = standard_dist[i] * standard_total_allocation / 100
            
            # Add epsilon allocation for new subject lines
            for sl_idx, alloc in epsilon_allocation.items():
                current_dist[sl_idx] = alloc
            
            # Ensure zero distribution for SLs that are supposed to have zero
            for sl_idx in zero_dist_sls:
                current_dist[sl_idx] = 0
            
            # Distribute sends according to the current distribution
            for i in range(max_subjectlines):
                if i in active_sl_indices and i not in zero_dist_sls:
                    sends_arr[i] = round(current_dist[i] * week_total_sends)
                    opens_arr[i] = round(np.random.uniform(
                        sl_open_rate_ranges[all_sls[i]][0], 
                        sl_open_rate_ranges[all_sls[i]][1]
                    ) * sends_arr[i]) if sends_arr[i] > 0 else 0
            
            # Update cumulative tracking
            for i in range(max_subjectlines):
                cumulative_sends[i] += sends_arr[i]
                cumulative_opens[i] += opens_arr[i]
            
            print(f"Sends this week: {[sends_arr[i] for i in active_sl_indices]}")
            print(f"Opens this week: {[opens_arr[i] for i in active_sl_indices]}")
            print(f"Cumulative sends: {[cumulative_sends[i] for i in active_sl_indices]}")
            print(f"Cumulative opens: {[cumulative_opens[i] for i in active_sl_indices]}")
            
            # Run MAB for next week's distribution
            # First, get non-zero distribution active subject lines
            active_non_zero_sls = [i for i in active_sl_indices if i not in zero_dist_sls]
            
            if active_non_zero_sls:
                active_sends = [cumulative_sends[i] for i in active_non_zero_sls]
                active_opens = [cumulative_opens[i] for i in active_non_zero_sls]
                
                next_dist_filtered = run_ts_mab(active_sends, active_opens)
                
                # Create full distribution array
                next_dist = [0] * max_subjectlines
                for i, sl_idx in enumerate(active_non_zero_sls):
                    next_dist[sl_idx] = next_dist_filtered[i]
                
                # Check if any SL has 100% distribution
                if 100 in next_dist:
                    winner_idx = next_dist.index(100)
                    print(f"Winner found! {all_sls[winner_idx]} has 100% distribution.")
                    print(f"Simulation ending early at week {week}.")
                    # Will exit loop after this iteration
            else:
                # No active subject lines with non-zero distribution
                next_dist = [0] * max_subjectlines
            
            print(f"Next distribution: {[next_dist[i] for i in active_sl_indices]}")
            
            # Store data for this week
            all_weeks.append(week)
            all_total_sends.append(week_total_sends)
            all_sl_distributions.append(current_dist.copy())
            all_sl_sends.append(sends_arr)
            all_sl_opens.append(opens_arr)
            all_cumulative_sends.append(cumulative_sends.copy())
            all_cumulative_opens.append(cumulative_opens.copy())
            
            # Identify new zero distribution SLs for the next week (these will be used to add new SLs)
            # Only consider active SLs that had non-zero distribution this week but will have zero next week
            new_zero_dist_sls = set([i for i in active_sl_indices 
                                   if i not in zero_dist_sls and next_dist[i] == 0])
            
            # Update overall zero_dist_sls
            zero_dist_sls.update(new_zero_dist_sls)
            
            if new_zero_dist_sls:
                print(f"Subject lines with zero distribution for next week: {[all_sls[i] for i in new_zero_dist_sls]}")
                
            # Check if we need to end the simulation early
            if 100 in next_dist:
                # We'll exit after this iteration
                break
    
    # Create comprehensive results DataFrame
    results = []
    
    for i in range(len(all_weeks)):
        week_data = {
            'week': all_weeks[i],
            'total_sends': all_total_sends[i],
            'active_sls': ','.join([all_sls[idx] for idx in all_active_sls[i]])
        }
        
        for j in range(max_subjectlines):
            sl_name = all_sls[j]
            week_data[f'{sl_name}_status'] = all_sl_statuses[i][j]
            week_data[f'{sl_name}_distribution'] = all_sl_distributions[i][j]
            week_data[f'{sl_name}_sends'] = all_sl_sends[i][j]
            week_data[f'{sl_name}_opens'] = all_sl_opens[i][j]
            week_data[f'{sl_name}_cumulative_sends'] = all_cumulative_sends[i][j]
            week_data[f'{sl_name}_cumulative_opens'] = all_cumulative_opens[i][j]
            
            # Calculate CTR
            if all_sl_sends[i][j] > 0:
                week_data[f'{sl_name}_weekly_ctr'] = all_sl_opens[i][j] / all_sl_sends[i][j] * 100
            else:
                week_data[f'{sl_name}_weekly_ctr'] = 0
                
            if all_cumulative_sends[i][j] > 0:
                week_data[f'{sl_name}_cumulative_ctr'] = all_cumulative_opens[i][j] / all_cumulative_sends[i][j] * 100
            else:
                week_data[f'{sl_name}_cumulative_ctr'] = 0
        
        results.append(week_data)
    
    # Add next week's predicted distribution if simulation didn't end with a winner
    if len(all_weeks) == n_weeks:
        next_week_data = {'week': n_weeks + 1, 'total_sends': 0}
        
        # Determine active SLs for the prediction
        active_sls_for_prediction = [i for i in active_sl_indices if i not in zero_dist_sls]
        next_week_data['active_sls'] = ','.join([all_sls[idx] for idx in active_sls_for_prediction])
        
        for j in range(max_subjectlines):
            sl_name = all_sls[j]
            
            if j in active_sl_indices:
                # Carry forward the status from the last week
                next_week_data[f'{sl_name}_status'] = all_sl_statuses[-1][j]
                
                # Prediction only makes sense for active SLs
                if j in active_sls_for_prediction:
                    next_week_data[f'{sl_name}_distribution'] = next_dist[j]
                else:
                    next_week_data[f'{sl_name}_distribution'] = 0
            else:
                next_week_data[f'{sl_name}_status'] = "inactive"
                next_week_data[f'{sl_name}_distribution'] = 0
                
            next_week_data[f'{sl_name}_sends'] = 0
            next_week_data[f'{sl_name}_opens'] = 0
            next_week_data[f'{sl_name}_cumulative_sends'] = cumulative_sends[j]
            next_week_data[f'{sl_name}_cumulative_opens'] = cumulative_opens[j]
            next_week_data[f'{sl_name}_weekly_ctr'] = 0
            
            if cumulative_sends[j] > 0:
                next_week_data[f'{sl_name}_cumulative_ctr'] = cumulative_opens[j] / cumulative_sends[j] * 100
            else:
                next_week_data[f'{sl_name}_cumulative_ctr'] = 0
        
        results.append(next_week_data)
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    result_df = generate_weekly_sends(n_weeks=5, n_subjectlines=5, max_subjectlines=10, epsilon=0.2, new_sl_weeks=2)
    print("\nSimulation Results DataFrame:")
    print(result_df)
