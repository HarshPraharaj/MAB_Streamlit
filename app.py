import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import generate_weekly_sends
import time
from scipy import stats


def aggregate_simulation_results(simulation_results, confidence_level=0.95):
    """
    Aggregate data from multiple simulation runs and calculate statistics.
    
    Parameters:
    -----------
    simulation_results : list
        List of DataFrames containing results from each simulation run
    confidence_level : float, default=0.95
        Confidence level for calculating confidence intervals
    
    Returns:
    --------
    dict
        Dictionary containing aggregated statistics
    """
    if not simulation_results:
        return None
    
    # Extract key statistics from each simulation
    n_simulations = len(simulation_results)
    
    # Initialize dictionaries to store aggregated data
    agg_data = {
        'weekly_distributions': {},  # SL -> week -> [distribution values]
        'weekly_ctrs': {},  # SL -> week -> [ctr values]
        'cumulative_ctrs': {},  # SL -> week -> [cumulative ctr values]
        'final_distributions': {},  # SL -> [final distribution values]
        'final_ctrs': {},  # SL -> [final ctr values]
        'simulation_durations': [],  # List of simulation durations (weeks)
        'exploration_costs': [],  # List of exploration costs
        'exploration_percentages': [],  # List of exploration cost percentages
        'initial_subject_lines': set(),  # Set of initial subject lines
        'new_subject_lines': set(),  # Set of new subject lines introduced
        'winners': {},  # SL -> count of times it was the winner
    }
    
    # Helper function to extract subject line names from active_sls string
    def extract_sls(active_sls):
        return active_sls.split(',') if isinstance(active_sls, str) else []
    
    # Helper function to calculate statistics
    def calculate_stats(values):
        values = np.array(values)
        mean = np.mean(values)
        
        # Calculate confidence interval
        if len(values) > 1:
            confidence = confidence_level
            z = 1.96  # for 95% confidence
            if confidence == 0.90:
                z = 1.645
            elif confidence == 0.99:
                z = 2.576
                
            std_err = np.std(values, ddof=1) / np.sqrt(len(values))
            margin = z * std_err
            ci_lower = mean - margin
            ci_upper = mean + margin
        else:
            ci_lower = mean
            ci_upper = mean
            
        return {
            'mean': mean,
            'std': np.std(values) if len(values) > 1 else 0,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'min': np.min(values) if len(values) > 0 else 0,
            'max': np.max(values) if len(values) > 0 else 0,
            'values': values.tolist()
        }
    
    # Process each simulation
    for i, df in enumerate(simulation_results):
        # Get all subject lines that appeared in the simulation
        all_sls = set()
        for week_row in df.iterrows():
            active_sl_str = week_row[1].get('active_sls', '')
            all_sls.update(extract_sls(active_sl_str))
        
        # Get initial subject lines (week 1)
        if not df.empty:
            first_week = df[df['week'] == 1]
            if not first_week.empty:
                active_sls_first_week = extract_sls(first_week.iloc[0].get('active_sls', ''))
                agg_data['initial_subject_lines'].update(active_sls_first_week)
                
                # Any subject line not in first week is a new subject line
                for sl in all_sls:
                    if sl not in active_sls_first_week:
                        agg_data['new_subject_lines'].add(sl)
        
        # Process weekly distributions and CTRs
        for sl in all_sls:
            # Initialize if not yet in the dictionaries
            if sl not in agg_data['weekly_distributions']:
                agg_data['weekly_distributions'][sl] = {}
                agg_data['weekly_ctrs'][sl] = {}
                agg_data['cumulative_ctrs'][sl] = {}
                agg_data['final_distributions'][sl] = []
                agg_data['final_ctrs'][sl] = []
            
            for week_num in df['week'].unique():
                week_data = df[df['week'] == week_num]
                if not week_data.empty:
                    # Distribution
                    if f'{sl}_distribution' in week_data.columns:
                        dist_value = week_data.iloc[0][f'{sl}_distribution']
                        if week_num not in agg_data['weekly_distributions'][sl]:
                            agg_data['weekly_distributions'][sl][week_num] = []
                        agg_data['weekly_distributions'][sl][week_num].append(dist_value)
                    
                    # Weekly CTR
                    if f'{sl}_weekly_ctr' in week_data.columns:
                        ctr_value = week_data.iloc[0][f'{sl}_weekly_ctr']
                        if week_num not in agg_data['weekly_ctrs'][sl]:
                            agg_data['weekly_ctrs'][sl][week_num] = []
                        agg_data['weekly_ctrs'][sl][week_num].append(ctr_value)
                    
                    # Cumulative CTR
                    if f'{sl}_cumulative_ctr' in week_data.columns:
                        cum_ctr_value = week_data.iloc[0][f'{sl}_cumulative_ctr']
                        if week_num not in agg_data['cumulative_ctrs'][sl]:
                            agg_data['cumulative_ctrs'][sl][week_num] = []
                        agg_data['cumulative_ctrs'][sl][week_num].append(cum_ctr_value)
        
        # Calculate simulation duration
        duration = max(df['week']) if not df.empty else 0
        agg_data['simulation_durations'].append(duration)
        
        # Get final distributions and CTRs
        last_week = df[df['week'] == duration]
        if not last_week.empty:
            for sl in all_sls:
                if f'{sl}_distribution' in last_week.columns:
                    # Use the distribution that was active during the final week, not the prediction for the next week
                    final_dist = last_week.iloc[0][f'{sl}_distribution']
                    agg_data['final_distributions'][sl].append(final_dist)
                    
                    # Check if this SL was the winner (100% distribution)
                    if final_dist == 100:
                        if sl not in agg_data['winners']:
                            agg_data['winners'][sl] = 0
                        agg_data['winners'][sl] += 1
                
                if f'{sl}_cumulative_ctr' in last_week.columns:
                    final_ctr = last_week.iloc[0][f'{sl}_cumulative_ctr']
                    agg_data['final_ctrs'][sl].append(final_ctr)
        
        # Calculate exploration cost
        if not df.empty:
            last_week_data = df[df['week'] == duration].iloc[0]
            
            # Get initial subject lines
            initial_sls = list(agg_data['initial_subject_lines'])
            new_sls = list(agg_data['new_subject_lines'])
            
            # Find the best initial subject line CTR
            best_initial_sl_ctr = 0
            best_initial_sl = None
            
            for sl in initial_sls:
                if f'{sl}_cumulative_ctr' in last_week_data:
                    ctr = last_week_data[f'{sl}_cumulative_ctr']
                    if ctr > best_initial_sl_ctr:
                        best_initial_sl_ctr = ctr
                        best_initial_sl = sl
            
            # Calculate exploration cost
            if best_initial_sl and best_initial_sl_ctr > 0:
                actual_opens = 0
                potential_opens = 0
                
                for sl in new_sls:
                    if f'{sl}_cumulative_sends' in last_week_data and f'{sl}_cumulative_opens' in last_week_data:
                        sends = last_week_data[f'{sl}_cumulative_sends']
                        opens = last_week_data[f'{sl}_cumulative_opens']
                        
                        actual_opens += opens
                        potential_opens += sends * (best_initial_sl_ctr / 100)
                
                exploration_cost = max(0, potential_opens - actual_opens)
                agg_data['exploration_costs'].append(exploration_cost)
                
                # Calculate total opens
                total_opens = 0
                for sl in all_sls:
                    if f'{sl}_cumulative_opens' in last_week_data:
                        total_opens += last_week_data[f'{sl}_cumulative_opens']
                
                if total_opens > 0:
                    exploration_percentage = exploration_cost / total_opens * 100
                    agg_data['exploration_percentages'].append(exploration_percentage)
    
    # Calculate statistics for each aggregated metric
    aggregate_stats = {
        'n_simulations': n_simulations,
        'simulation_duration': calculate_stats(agg_data['simulation_durations']),
        'initial_subject_lines': list(agg_data['initial_subject_lines']),
        'new_subject_lines': list(agg_data['new_subject_lines']),
        'exploration_cost': calculate_stats(agg_data['exploration_costs']),
        'exploration_percentage': calculate_stats(agg_data['exploration_percentages']),
        'winners': {sl: {'count': count, 'percentage': count / n_simulations * 100} 
                   for sl, count in agg_data['winners'].items()},
        'weekly_distributions': {},
        'weekly_ctrs': {},
        'cumulative_ctrs': {},
        'final_distributions': {},
        'final_ctrs': {}
    }
    
    # Calculate statistics for weekly metrics
    for sl in agg_data['weekly_distributions']:
        aggregate_stats['weekly_distributions'][sl] = {}
        for week in agg_data['weekly_distributions'][sl]:
            values = agg_data['weekly_distributions'][sl][week]
            aggregate_stats['weekly_distributions'][sl][week] = calculate_stats(values)
    
    for sl in agg_data['weekly_ctrs']:
        aggregate_stats['weekly_ctrs'][sl] = {}
        for week in agg_data['weekly_ctrs'][sl]:
            values = agg_data['weekly_ctrs'][sl][week]
            aggregate_stats['weekly_ctrs'][sl][week] = calculate_stats(values)
    
    for sl in agg_data['cumulative_ctrs']:
        aggregate_stats['cumulative_ctrs'][sl] = {}
        for week in agg_data['cumulative_ctrs'][sl]:
            values = agg_data['cumulative_ctrs'][sl][week]
            aggregate_stats['cumulative_ctrs'][sl][week] = calculate_stats(values)
    
    # Calculate statistics for final metrics
    for sl in agg_data['final_distributions']:
        values = agg_data['final_distributions'][sl]
        aggregate_stats['final_distributions'][sl] = calculate_stats(values)
    
    for sl in agg_data['final_ctrs']:
        values = agg_data['final_ctrs'][sl]
        aggregate_stats['final_ctrs'][sl] = calculate_stats(values)
    
    return aggregate_stats


def calculate_confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for a list of values."""
    if len(data) < 2:
        return data[0] if data else 0, data[0] if data else 0
    
    mean = np.mean(data)
    
    # Set z-score based on confidence level
    z = 1.96  # for 95% confidence
    if confidence == 0.90:
        z = 1.645
    elif confidence == 0.99:
        z = 2.576
        
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    margin = z * std_err
    
    return mean - margin, mean + margin


# Set page configuration
st.set_page_config(
    page_title="MAB Simulation Dashboard",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("Multi-Armed Bandit Simulation Dashboard")
st.markdown("""
This application simulates Thompson Sampling Multi-Armed Bandit (TS-MAB) experiments over multiple weeks.
You can adjust parameters and see the results visualized in various ways.
""")

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

n_weeks = st.sidebar.slider("Number of Weeks", min_value=2, max_value=20, value=10, step=1)
n_initial_subjectlines = st.sidebar.slider("Initial Subject Lines", min_value=5, max_value=10, value=5, step=1)
max_subjectlines = st.sidebar.slider("Maximum Subject Lines", min_value=n_initial_subjectlines, max_value=20, value=15, step=1)
epsilon = st.sidebar.slider("Epsilon (New SL Allocation)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
new_sl_weeks = st.sidebar.slider("New SL Weeks", min_value=1, max_value=5, value=2, step=1)

# Real campaign simulation option
st.sidebar.header("Campaign Simulation Type")
real_campaign = st.sidebar.checkbox("Real Campaign Simulation", value=False)
if real_campaign:
    st.sidebar.info("""
    In real campaign simulation:
    1. You set open rates for initial subject lines only
    2. New subject lines will sample open rates from initial subject lines
    3. This simulates real campaigns where new content performs similar to existing content
    """)
    
    # Add option for a specific new subject line with controlled performance
    include_special_sl = st.sidebar.checkbox("Include Special Subject Line", value=False)
    if include_special_sl:
        special_sl_std_dev = st.sidebar.slider(
            "Special SL Performance (std dev from mean)", 
            min_value=-10.0, 
            max_value=10.0, 
            value=1.0, 
            step=0.1,
            help="Set how many standard deviations from the mean the special subject line's open rate should be. Positive values create better-performing subject lines, negative values create worse-performing ones."
        )
        
        st.sidebar.info(f"""
        A special subject line will be introduced with an open rate approximately {special_sl_std_dev:.1f} standard deviations 
        from the mean of initial subject lines.
        
        - Positive values mean better performance than average
        - Negative values mean worse performance than average
        - ±1.0 covers ~68% of normal distribution
        - ±2.0 covers ~95% of normal distribution
        - ±3.0 covers ~99.7% of normal distribution
        - Values beyond ±3.0 represent extremely rare performance (less than 0.3% likelihood)
        """)
    else:
        include_special_sl = False
        special_sl_std_dev = 0.0

# Add parameter for multiple runs
st.sidebar.header("Monte Carlo Settings")
run_monte_carlo = st.sidebar.checkbox("Run Multiple Simulations", value=False)
if run_monte_carlo:
    n_simulations = st.sidebar.slider("Number of Simulations", min_value=10, max_value=100, value=30, step=10)
    confidence_level = st.sidebar.slider("Confidence Level (%)", min_value=80, max_value=99, value=95, step=1)
    st.sidebar.info("Running multiple simulations will take longer but provide statistical confidence intervals.")
else:
    n_simulations = 1
    confidence_level = 95

# Open Rate Settings
st.sidebar.header("Subject Line Open Rates")
use_custom_rates = st.sidebar.checkbox("Customize Open Rates", value=False)

# Default presets for open rates
default_min_rates = {
    "SL_1": 0.46, "SL_2": 0.39, "SL_3": 0.43, "SL_4": 0.37, "SL_5": 0.39,
    "SL_6": 0.47, "SL_7": 0.53, "SL_8": 0.46, "SL_9": 0.54, "SL_10": 0.51,
    "SL_11": 0.49, "SL_12": 0.56, "SL_13": 0.47, "SL_14": 0.53, "SL_15": 0.48,
    "SL_16": 0.52, "SL_17": 0.50, "SL_18": 0.55, "SL_19": 0.49, "SL_20": 0.51
}

default_max_rates = {
    "SL_1": 0.55, "SL_2": 0.40, "SL_3": 0.50, "SL_4": 0.38, "SL_5": 0.40,
    "SL_6": 0.62, "SL_7": 0.68, "SL_8": 0.61, "SL_9": 0.69, "SL_10": 0.66,
    "SL_11": 0.64, "SL_12": 0.70, "SL_13": 0.62, "SL_14": 0.68, "SL_15": 0.63,
    "SL_16": 0.67, "SL_17": 0.65, "SL_18": 0.71, "SL_19": 0.64, "SL_20": 0.66
}

# Dictionary to store user-defined open rates
custom_min_rates = {}
custom_max_rates = {}

if use_custom_rates:
    st.sidebar.info("Set min and max open rates for each subject line (as decimal: 0.15 = 15%)")
    
    # Create expandable sections for groups of subject lines
    with st.sidebar.expander("Initial Subject Lines"):
        for i in range(1, n_initial_subjectlines + 1):
            sl_name = f"SL_{i}"
            cols = st.columns(2)
            with cols[0]:
                min_rate = st.number_input(f"{sl_name} Min", min_value=0.01, max_value=0.95, value=default_min_rates[sl_name], step=0.01, format="%.2f")
            with cols[1]:
                max_rate = st.number_input(f"{sl_name} Max", min_value=min_rate, max_value=0.95, value=max(min_rate+0.05, default_max_rates[sl_name]), step=0.01, format="%.2f")
            custom_min_rates[sl_name] = min_rate
            custom_max_rates[sl_name] = max_rate
    
    # Only show additional subject lines if not in real campaign mode
    if not real_campaign:
        with st.sidebar.expander("Additional Subject Lines"):
            for i in range(n_initial_subjectlines + 1, max_subjectlines + 1):
                sl_name = f"SL_{i}"
                cols = st.columns(2)
                with cols[0]:
                    min_rate = st.number_input(f"{sl_name} Min", min_value=0.01, max_value=0.95, value=default_min_rates[sl_name], step=0.01, format="%.2f")
                with cols[1]:
                    max_rate = st.number_input(f"{sl_name} Max", min_value=min_rate, max_value=0.95, value=max(min_rate+0.05, default_max_rates[sl_name]), step=0.01, format="%.2f")
                custom_min_rates[sl_name] = min_rate
                custom_max_rates[sl_name] = max_rate
    elif real_campaign:
        st.sidebar.info("In real campaign mode, open rates for new subject lines will be sampled from the initial subject lines. No need to set them manually.")

# Run simulation button
if st.sidebar.button("Run Simulation"):
    with st.spinner(f"Running {'multiple simulations' if run_monte_carlo else 'simulation'}..."):
        # Create containers to capture the print outputs
        simulation_output = st.empty()
        
        # Redirect stdout to capture print statements
        import io
        import contextlib
        import sys
        
        # Create a StringIO object to capture output
        f = io.StringIO()
        
        # Lists to store results from multiple simulations
        all_simulation_results = []
        
        with contextlib.redirect_stdout(f):
            if run_monte_carlo:
                progress_bar = st.progress(0)
                for sim_idx in range(n_simulations):
                    # Update progress bar
                    progress_bar.progress((sim_idx + 1) / n_simulations)
                    
                    # Run a single simulation
                    result_df = generate_weekly_sends(
                        n_weeks=n_weeks, 
                        n_subjectlines=n_initial_subjectlines, 
                        max_subjectlines=max_subjectlines,
                        epsilon=epsilon,
                        new_sl_weeks=new_sl_weeks,
                        custom_min_rates=custom_min_rates if use_custom_rates else None,
                        custom_max_rates=custom_max_rates if use_custom_rates else None,
                        real_campaign=real_campaign,
                        include_special_sl=include_special_sl,
                        special_sl_std_dev=special_sl_std_dev,
                        verbose=(sim_idx == 0)  # Only show verbose output for first run
                    )
                    all_simulation_results.append(result_df)
                    
                # Aggregate results from all simulations
                result_df = all_simulation_results[0]  # Use the first run for primary display
                aggregate_results = aggregate_simulation_results(all_simulation_results, confidence_level/100)
                
                # Show completion message
                progress_bar.empty()
                st.success(f"Completed {n_simulations} simulations!")
            else:
                # Just run a single simulation
                result_df = generate_weekly_sends(
                    n_weeks=n_weeks, 
                    n_subjectlines=n_initial_subjectlines, 
                    max_subjectlines=max_subjectlines,
                    epsilon=epsilon,
                    new_sl_weeks=new_sl_weeks,
                    custom_min_rates=custom_min_rates if use_custom_rates else None,
                    custom_max_rates=custom_max_rates if use_custom_rates else None,
                    real_campaign=real_campaign,
                    include_special_sl=include_special_sl,
                    special_sl_std_dev=special_sl_std_dev,
                    verbose=True
                )
                all_simulation_results = [result_df]
                aggregate_results = None
        
        # Display the captured output
        simulation_log = f.getvalue()
        
        # Identify all subject lines that were used
        used_sls = []
        for idx, row in result_df.iterrows():
            if 'active_sls' in row:
                sls = row['active_sls'].split(',')
                for sl in sls:
                    if sl and sl not in used_sls:
                        used_sls.append(sl)
        
        # Create data for visualizations
        weekly_dists = {}
        weekly_ctrs = {}
        weekly_ctr_df = pd.DataFrame(index=result_df['week'])
        cumulative_ctr_df = pd.DataFrame(index=result_df['week'])
        
        for sl in used_sls:
            if f'{sl}_distribution' in result_df.columns:
                weekly_dists[sl] = result_df[f'{sl}_distribution']
            
            if f'{sl}_weekly_ctr' in result_df.columns:
                weekly_ctrs[sl] = result_df[f'{sl}_weekly_ctr']
                weekly_ctr_df[sl] = result_df[f'{sl}_weekly_ctr']
                cumulative_ctr_df[sl] = result_df[f'{sl}_cumulative_ctr']
        
        # Create main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Simulation Log", "Summary", "Metrics", "Visualizations", "Raw Data"])
        
        # Simulation Log Tab
        with tab1:
            st.subheader("Simulation Log")
            st.text(simulation_log)
        
        # Summary Tab  
        with tab2:
            st.subheader("Simulation Summary")
            
            if run_monte_carlo:
                st.info(f"Results aggregated over {n_simulations} simulations with {confidence_level}% confidence intervals.")
            
            # Display final week metrics
            st.subheader("Final Week Metrics")
            
            # Get last week data (excluding the prediction row if it exists)
            last_week_data = result_df.iloc[-2] if len(result_df) > n_weeks else result_df.iloc[-1]
            
            # Create columns to display metrics
            cols = st.columns(4)
            with cols[0]:
                st.metric("Total Weeks", max(result_df['week']))
            with cols[1]:
                total_sends = sum(result_df['total_sends'])
                st.metric("Total Sends", f"{total_sends:,}")
            with cols[2]:
                # Calculate total opens across all subject lines
                total_opens = 0
                for sl in used_sls:
                    if f'{sl}_cumulative_opens' in last_week_data:
                        total_opens += last_week_data[f'{sl}_cumulative_opens']
                st.metric("Total Opens", f"{int(total_opens):,}")
            with cols[3]:
                # Calculate overall CTR
                overall_ctr = (total_opens / total_sends * 100) if total_sends > 0 else 0
                st.metric("Overall CTR", f"{overall_ctr:.2f}%")
            
            # Display campaign improvement metrics for single simulation
            if not run_monte_carlo and 'mab_benefit' in last_week_data:
                st.subheader("Campaign Improvement Summary")
                
                # Show improvement between initial subject lines and overall campaign
                improvement_cols = st.columns(3)
                with improvement_cols[0]:
                    st.metric("Initial Subject Lines CTR", 
                             f"{last_week_data['cumulative_initial_sl_ctr']:.2f}%")
                
                with improvement_cols[1]:
                    st.metric("Overall Campaign CTR", 
                             f"{last_week_data['cumulative_campaign_ctr']:.2f}%",
                             delta=f"{last_week_data['cumulative_campaign_ctr'] - last_week_data['cumulative_initial_sl_ctr']:.2f}%")
                
                with improvement_cols[2]:
                    st.metric("Improvement with MAB", 
                             f"{last_week_data['mab_benefit']:.2f}%")
                
                # Additional context
                if 'best_initial_sl' in last_week_data and 'num_new_active_sls' in last_week_data:
                    st.info(f"The simulation added {int(last_week_data['num_new_active_sls'])} new subject lines, " +
                           f"improving over the best initial subject line ({last_week_data['best_initial_sl']}) " +
                           f"with a CTR of {last_week_data['best_initial_sl_ctr']:.2f}%.")
            
            # Monte Carlo Results
            if run_monte_carlo and aggregate_results:
                st.subheader("Aggregate Performance Metrics")
                
                # Display aggregate metrics with confidence intervals
                agg_cols = st.columns(3)
                with agg_cols[0]:
                    if 'simulation_duration' in aggregate_results:
                        mean_weeks = aggregate_results['simulation_duration']['mean']
                        ci_low = aggregate_results['simulation_duration']['ci_lower']
                        ci_high = aggregate_results['simulation_duration']['ci_upper']
                        st.metric("Avg Simulation Duration", f"{mean_weeks:.1f} weeks")
                        st.caption(f"95% CI: [{ci_low:.1f}, {ci_high:.1f}]")
                    else:
                        st.metric("Avg Simulation Duration", "N/A")
                    
                with agg_cols[1]:
                    # Calculate overall CTR across all simulations
                    overall_ctrs = []
                    for sim_df in all_simulation_results:
                        if not sim_df.empty:
                            total_sends = sim_df['total_sends'].sum()
                            total_opens = 0
                            
                            # Get the last week data
                            last_week = sim_df.iloc[-2] if len(sim_df) > n_weeks else sim_df.iloc[-1]
                            
                            # Sum up all opens
                            for sl in used_sls:
                                if f'{sl}_cumulative_opens' in last_week:
                                    total_opens += last_week[f'{sl}_cumulative_opens']
                            
                            if total_sends > 0:
                                overall_ctrs.append(total_opens / total_sends * 100)
                    
                    if overall_ctrs:
                        mean_ctr = np.mean(overall_ctrs)
                        ci_low, ci_high = calculate_confidence_interval(overall_ctrs, confidence_level/100)
                        st.metric("Avg Overall CTR", f"{mean_ctr:.2f}%")
                        st.caption(f"95% CI: [{ci_low:.2f}%, {ci_high:.2f}%]")
                    else:
                        st.metric("Avg Overall CTR", "N/A")
                    
                with agg_cols[2]:
                    if 'new_subject_lines' in aggregate_results:
                        mean_new_sls = len(aggregate_results['new_subject_lines'])
                        st.metric("New Subject Lines", f"{mean_new_sls}")
                    else:
                        st.metric("New Subject Lines", "N/A")
                
                # Display aggregate subject line metrics
                st.subheader("Subject Line Performance")
                st.caption("Shows metrics from the final week of each simulation (not predictions for future weeks)")
                
                # Create a formatted dataframe for display
                sl_results = []
                
                for sl in used_sls:
                    if sl in aggregate_results['final_ctrs']:
                        ctr_stats = aggregate_results['final_ctrs'][sl]
                        mean_ctr = ctr_stats['mean']
                        ci_low_ctr = ctr_stats['ci_lower']
                        ci_high_ctr = ctr_stats['ci_upper']
                        
                        dist_stats = aggregate_results['final_distributions'][sl] if sl in aggregate_results['final_distributions'] else None
                        mean_dist = dist_stats['mean'] * 100 if dist_stats else 0  # Convert to percentage
                        
                        # Calculate winner frequency
                        winner_freq = 0
                        if 'winners' in aggregate_results and sl in aggregate_results['winners']:
                            winner_freq = aggregate_results['winners'][sl]['percentage']
                        
                        sl_results.append({
                            'Subject Line': sl,
                            'Final CTR': f"{mean_ctr:.2f}%",
                            '95% CI': f"[{ci_low_ctr:.2f}%, {ci_high_ctr:.2f}%]",
                            'Avg Final Distribution': f"{mean_dist:.2f}%",
                            'Winner Frequency': f"{winner_freq:.1f}%"
                        })
                
                if sl_results:
                    sl_results_df = pd.DataFrame(sl_results)
                    st.dataframe(sl_results_df)
                else:
                    st.info("No subject line statistics available.")
                
                # Plot winner distribution
                st.subheader("Winner Distribution")
                if 'winners' in aggregate_results and aggregate_results['winners']:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Sort winners by frequency
                    sorted_winners = sorted(aggregate_results['winners'].items(), 
                                           key=lambda x: x[1]['percentage'], 
                                           reverse=True)
                    
                    sls = [winner[0] for winner in sorted_winners]
                    freqs = [winner[1]['percentage'] for winner in sorted_winners]
                    
                    bars = ax.bar(sls, freqs)
                    ax.set_xlabel('Subject Line')
                    ax.set_ylabel('Win Percentage (%)')
                    ax.set_ylim(0, 100)
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.1f}%',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom')
                    
                    plt.title('Percentage of Simulations Where Subject Line Was the Winner')
                    plt.grid(axis='y', alpha=0.3)
                    st.pyplot(fig)
                else:
                    st.info("No clear winners found across simulations.")
            
            # Subject Line Introduction
            st.subheader("Subject Line Introduction Timeline")
            
            # Create a timeline of when each subject line was introduced
            intro_weeks = {}
            for sl in used_sls:
                for idx, row in result_df.iterrows():
                    if 'active_sls' in row and sl in row['active_sls'].split(','):
                        intro_weeks[sl] = row['week']
                        break
            
            # Sort by introduction week
            sorted_sls = sorted(intro_weeks.items(), key=lambda x: x[1])
            
            # Display as a table
            intro_data = []
            for sl, week in sorted_sls:
                # Get distribution for the final week (not prediction for next week)
                final_dist = last_week_data[f'{sl}_distribution'] if f'{sl}_distribution' in last_week_data else 0
                # Get final CTR
                final_ctr = last_week_data[f'{sl}_cumulative_ctr'] if f'{sl}_cumulative_ctr' in last_week_data else 0
                # Get final status
                final_status = last_week_data[f'{sl}_status'] if f'{sl}_status' in last_week_data else "unknown"
                
                intro_data.append({
                    "Subject Line": sl,
                    "Introduced in Week": week,
                    "Final Status": final_status.capitalize(),
                    "Final Distribution": f"{final_dist:.2%}" if final_dist > 0 else "0%",
                    "Final CTR": f"{final_ctr:.2f}%" if final_ctr > 0 else "0%"
                })
            
            intro_df = pd.DataFrame(intro_data)
            st.dataframe(intro_df)
        
        # Metrics Tab
        with tab3:
            # Display campaign improvement metrics for single simulation
            if not run_monte_carlo:
                st.subheader("Campaign Improvement Metrics")
                
                # Create a plot showing the progression of different metrics
                if 'cumulative_campaign_ctr' in result_df.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Add lines for each metric
                    if 'cumulative_campaign_ctr' in result_df.columns:
                        ax.plot(result_df['week'], result_df['cumulative_campaign_ctr'], 
                                marker='o', label='Overall Campaign CTR', linewidth=2)
                    
                    if 'cumulative_initial_sl_ctr' in result_df.columns:
                        ax.plot(result_df['week'], result_df['cumulative_initial_sl_ctr'], 
                                marker='s', label='Initial Subject Lines CTR', linewidth=2)
                    
                    if 'counterfactual_ctr' in result_df.columns:
                        ax.plot(result_df['week'], result_df['counterfactual_ctr'], 
                                marker='^', linestyle='--', label='Best Initial SL Only (counterfactual)', linewidth=2)
                    
                    # Add number of new subject lines as annotations
                    for i, row in result_df.iterrows():
                        if 'num_new_active_sls' in row and row['num_new_active_sls'] > 0:
                            ax.annotate(f"{int(row['num_new_active_sls'])} new SLs", 
                                       (row['week'], row['cumulative_campaign_ctr']),
                                       textcoords="offset points",
                                       xytext=(0,10),
                                       ha='center')
                    
                    ax.set_xlabel('Week')
                    ax.set_ylabel('CTR (%)')
                    ax.set_title('Campaign CTR Improvement with New Subject Lines')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Create summary table
                    last_week_data = result_df.iloc[-2] if len(result_df) > n_weeks else result_df.iloc[-1]
                    
                    if 'mab_benefit' in last_week_data:
                        # Display key metrics to prove improvement
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Overall Campaign CTR", 
                                     f"{last_week_data['cumulative_campaign_ctr']:.2f}%")
                            
                        with cols[1]:
                            st.metric("Initial SLs Only CTR", 
                                     f"{last_week_data['cumulative_initial_sl_ctr']:.2f}%")
                            
                        with cols[2]:
                            st.metric("Improvement with New SLs", 
                                     f"{last_week_data['mab_benefit']:.2f}%",
                                     delta=f"{last_week_data['mab_benefit']:.2f}%")
                        
                        # Add explanation
                        st.markdown("""
                        **Key Findings:**
                        
                        The data shows that introducing new subject lines significantly improves campaign performance.
                        
                        1. The **Overall Campaign CTR** shows the actual CTR achieved with the multi-armed bandit approach.
                        2. The **Initial SLs Only CTR** shows what the CTR would have been using only the initial subject lines.
                        3. The **Improvement** metric quantifies the benefit of adding new subject lines to the campaign.
                        
                        This improvement demonstrates that the MAB approach effectively discovers better-performing subject lines
                        and allocates more sends to them, resulting in higher overall campaign performance.
                        """)
                        
                        # Display best subject line info
                        if 'best_initial_sl' in last_week_data:
                            st.write(f"**Best Initial Subject Line:** {last_week_data['best_initial_sl']} with CTR of {last_week_data['best_initial_sl_ctr']:.2f}%")
                
                else:
                    st.info("Run the simulation to see campaign improvement metrics.")
            
            st.subheader("Exploration Metrics")
            
            # Add expandable explanation section
            with st.expander("ℹ️ How are these metrics calculated?"):
                st.markdown("""
                ### Metrics Calculation Methodology
                
                This dashboard analyzes the trade-offs between exploration (trying new subject lines) and exploitation (using the best-performing ones).
                
                **Terminology:**
                - **Initial Subject Lines**: The first set of subject lines (SL_1 through SL_{n_initial}) tested in Week 1
                - **New Subject Lines**: Any subject lines introduced after Week 1 when existing subject lines reach zero distribution
                - **Best Initial Subject Line**: The initial subject line with the highest cumulative CTR
                
                **Open Rate Calculations:**
                1. **Actual Open Rate**: The open rate achieved in the simulation based on actual subject line performance
                2. **Counterfactual Open Rate**: The estimated open rate if we had only used the best initial subject line for all new subject line sends
                3. **Open Rate Difference**: The difference between counterfactual and actual open rates
                
                **Exploration Cost Calculation:**
                1. For each new subject line, we calculate:
                   - Actual Opens: The actual number of opens received
                   - Potential Opens: The opens we would have received if we used the best initial subject line instead
                   - The difference (if positive) is the exploration cost
                """)
            
            if run_monte_carlo and aggregate_results:
                st.info(f"Showing aggregate results from {n_simulations} simulations with {confidence_level}% confidence intervals")
                
                # 1. Calculate actual vs counterfactual open rates across all simulations
                actual_open_rates = []
                counterfactual_open_rates = []
                
                for sim_df in all_simulation_results:
                    # Get the last real week
                    duration = max(sim_df['week']) if not sim_df.empty else 0
                    last_week_data = sim_df[sim_df['week'] == duration].iloc[0]
                    
                    # Identify initial subject lines
                    initial_sls = []
                    if not sim_df.empty:
                        first_week = sim_df[sim_df['week'] == 1]
                        if not first_week.empty and 'active_sls' in first_week.iloc[0]:
                            initial_sls = first_week.iloc[0]['active_sls'].split(',')
                    
                    # Find best initial subject line
                    best_initial_sl = None
                    best_initial_ctr = 0
                    
                    for sl in initial_sls:
                        if f'{sl}_cumulative_ctr' in last_week_data:
                            ctr = last_week_data[f'{sl}_cumulative_ctr']
                            if ctr > best_initial_ctr:
                                best_initial_ctr = ctr
                                best_initial_sl = sl
                    
                    if best_initial_sl:
                        # Calculate actual total opens and sends
                        total_sends = 0
                        total_opens = 0
                        
                        # Identify new subject lines (not in initial set)
                        new_sls = [sl for sl in used_sls if sl not in initial_sls]
                        
                        # Calculate cumulative opens and sends
                        for sl in initial_sls + new_sls:
                            if f'{sl}_cumulative_sends' in last_week_data and f'{sl}_cumulative_opens' in last_week_data:
                                sl_sends = last_week_data[f'{sl}_cumulative_sends']
                                sl_opens = last_week_data[f'{sl}_cumulative_opens']
                                
                                total_sends += sl_sends
                                total_opens += sl_opens
                        
                        # Calculate counterfactual opens (if we had used best initial SL for all new SL sends)
                        counterfactual_opens = total_opens
                        
                        for sl in new_sls:
                            if f'{sl}_cumulative_sends' in last_week_data and f'{sl}_cumulative_opens' in last_week_data:
                                sl_sends = last_week_data[f'{sl}_cumulative_sends']
                                sl_opens = last_week_data[f'{sl}_cumulative_opens']
                                
                                # Replace with best initial SL performance
                                counterfactual_opens = counterfactual_opens - sl_opens + (sl_sends * best_initial_ctr / 100)
                        
                        # Calculate open rates
                        if total_sends > 0:
                            actual_open_rate = (total_opens / total_sends) * 100
                            counterfactual_open_rate = (counterfactual_opens / total_sends) * 100
                            
                            actual_open_rates.append(actual_open_rate)
                            counterfactual_open_rates.append(counterfactual_open_rate)
                
                # Calculate statistics for open rates
                if actual_open_rates and counterfactual_open_rates:
                    # Mean values
                    mean_actual = np.mean(actual_open_rates)
                    mean_counterfactual = np.mean(counterfactual_open_rates)
                    mean_difference = mean_counterfactual - mean_actual
                    
                    # Confidence intervals
                    actual_ci_low, actual_ci_high = calculate_confidence_interval(actual_open_rates, confidence_level/100)
                    counterfactual_ci_low, counterfactual_ci_high = calculate_confidence_interval(counterfactual_open_rates, confidence_level/100)
                    
                    # Calculate differences for confidence interval on difference
                    differences = [c - a for c, a in zip(counterfactual_open_rates, actual_open_rates)]
                    diff_ci_low, diff_ci_high = calculate_confidence_interval(differences, confidence_level/100)
                    
                    # Display results
                    st.subheader("Open Rate Analysis")
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Actual Open Rate", f"{mean_actual:.2f}%")
                        st.caption(f"{confidence_level}% CI: [{actual_ci_low:.2f}%, {actual_ci_high:.2f}%]")
                        st.markdown("*Based on the actual subject line performance*")
                    
                    with cols[1]:
                        st.metric("Counterfactual Open Rate", f"{mean_counterfactual:.2f}%")
                        st.caption(f"{confidence_level}% CI: [{counterfactual_ci_low:.2f}%, {counterfactual_ci_high:.2f}%]")
                        st.markdown("*If only used best initial subject line*")
                    
                    with cols[2]:
                        st.metric("Open Rate Difference", f"{mean_difference:.2f}%", 
                                 delta=f"{mean_difference:.2f}%" if mean_difference != 0 else None,
                                 delta_color="inverse")
                        st.caption(f"{confidence_level}% CI: [{diff_ci_low:.2f}%, {diff_ci_high:.2f}%]")
                        st.markdown("*Positive = exploration cost, Negative = exploration gain*")
                    
                    st.info(f"Across {n_simulations} simulations, exploring new subject lines {('decreased' if mean_difference > 0 else 'increased')} the overall open rate by {abs(mean_difference):.2f}% on average.")
                    
                    # Display individual simulation data
                    st.subheader("Simulation-level Open Rate Data")
                    
                    # Create dataframe with simulation results
                    sim_data = []
                    for i, (actual, counterfactual) in enumerate(zip(actual_open_rates, counterfactual_open_rates)):
                        diff = counterfactual - actual
                        sim_data.append({
                            "Simulation": i+1,
                            "Actual Open Rate (%)": f"{actual:.2f}",
                            "Counterfactual Open Rate (%)": f"{counterfactual:.2f}",
                            "Difference (%)": f"{diff:.2f}",
                            "Exploration Result": "Cost" if diff > 0 else "Gain"
                        })
                    
                    sim_df = pd.DataFrame(sim_data)
                    st.dataframe(sim_df)
                    
                    # Display exploration results summary
                    costs_count = sum(1 for d in differences if d > 0)
                    gains_count = sum(1 for d in differences if d <= 0)
                    
                    st.markdown(f"""
                    **Exploration Results Summary:**
                    - In {costs_count} simulations ({costs_count/n_simulations*100:.1f}%), exploration resulted in a cost (lower open rate)
                    - In {gains_count} simulations ({gains_count/n_simulations*100:.1f}%), exploration resulted in a gain (equal or higher open rate)
                    """)
                    
                    # Add exploration efficiency metrics
                    if 'exploration_cost' in aggregate_results and 'exploration_percentage' in aggregate_results:
                        st.subheader("Exploration Efficiency")
                        
                        cost_stats = aggregate_results['exploration_cost']
                        pct_stats = aggregate_results['exploration_percentage']
                        
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric("Avg Exploration Cost (Opens)", f"{int(cost_stats['mean']):,}")
                            st.caption(f"{confidence_level}% CI: [{int(cost_stats['ci_lower']):,}, {int(cost_stats['ci_upper']):,}]")
                        
                        with cols[1]:
                            st.metric("Avg Exploration Cost (%)", f"{pct_stats['mean']:.2f}%")
                            st.caption(f"{confidence_level}% CI: [{pct_stats['ci_lower']:.2f}%, {pct_stats['ci_upper']:.2f}%]")
                
            else:
                # Single simulation metrics
                if not result_df.empty:
                    # Get last week data (excluding prediction row)
                    last_week_data = result_df.iloc[-2] if len(result_df) > n_weeks else result_df.iloc[-1]
                    
                    # Identify initial subject lines
                    initial_sls = []
                    first_week = result_df[result_df['week'] == 1]
                    if not first_week.empty and 'active_sls' in first_week.iloc[0]:
                        initial_sls = first_week.iloc[0]['active_sls'].split(',')
                    
                    # New subject lines
                    new_sls = [sl for sl in used_sls if sl not in initial_sls]
                    
                    # Find best initial subject line
                    best_initial_sl = None
                    best_initial_ctr = 0
                    
                    for sl in initial_sls:
                        if f'{sl}_cumulative_ctr' in last_week_data:
                            ctr = last_week_data[f'{sl}_cumulative_ctr']
                            if ctr > best_initial_ctr:
                                best_initial_ctr = ctr
                                best_initial_sl = sl
                    
                    if best_initial_sl:
                        # Calculate actual total opens and sends
                        total_sends = 0
                        total_opens = 0
                        
                        # Calculate cumulative opens and sends
                        for sl in initial_sls + new_sls:
                            if f'{sl}_cumulative_sends' in last_week_data and f'{sl}_cumulative_opens' in last_week_data:
                                sl_sends = last_week_data[f'{sl}_cumulative_sends']
                                sl_opens = last_week_data[f'{sl}_cumulative_opens']
                                
                                total_sends += sl_sends
                                total_opens += sl_opens
                        
                        # Calculate counterfactual opens (if we had used best initial SL for all new SL sends)
                        counterfactual_opens = total_opens
                        
                        for sl in new_sls:
                            if f'{sl}_cumulative_sends' in last_week_data and f'{sl}_cumulative_opens' in last_week_data:
                                sl_sends = last_week_data[f'{sl}_cumulative_sends']
                                sl_opens = last_week_data[f'{sl}_cumulative_opens']
                                
                                # Replace with best initial SL performance
                                counterfactual_opens = counterfactual_opens - sl_opens + (sl_sends * best_initial_ctr / 100)
                        
                        # Calculate open rates
                        actual_open_rate = (total_opens / total_sends) * 100 if total_sends > 0 else 0
                        counterfactual_open_rate = (counterfactual_opens / total_sends) * 100 if total_sends > 0 else 0
                        difference = counterfactual_open_rate - actual_open_rate
                        
                        # Display results
                        st.subheader("Open Rate Analysis")
                        
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Actual Open Rate", f"{actual_open_rate:.2f}%")
                            st.markdown("*Based on the actual subject line performance*")
                        
                        with cols[1]:
                            st.metric("Counterfactual Open Rate", f"{counterfactual_open_rate:.2f}%")
                            st.markdown("*If only used best initial subject line*")
                        
                        with cols[2]:
                            st.metric("Open Rate Difference", f"{difference:.2f}%", 
                                     delta=f"{difference:.2f}%" if difference != 0 else None,
                                     delta_color="inverse")
                            st.markdown("*Positive = exploration cost, Negative = exploration gain*")
                        
                        # Determine if exploration was beneficial or costly
                        result_type = "cost" if difference > 0 else "gain"
                        
                        st.info(f"Exploring new subject lines {('decreased' if difference > 0 else 'increased')} the overall open rate by {abs(difference):.2f}%, which represents an exploration {result_type}.")
                        
                        # Detail breakdown for new subject lines
                        if new_sls:
                            st.subheader("New Subject Line Performance")
                            
                            new_sl_data = []
                            for sl in new_sls:
                                if f'{sl}_cumulative_sends' in last_week_data and f'{sl}_cumulative_opens' in last_week_data:
                                    sl_sends = last_week_data[f'{sl}_cumulative_sends']
                                    sl_opens = last_week_data[f'{sl}_cumulative_opens']
                                    
                                    # Calculate actual and counterfactual opens
                                    sl_actual_rate = (sl_opens / sl_sends) * 100 if sl_sends > 0 else 0
                                    sl_potential_opens = sl_sends * best_initial_ctr / 100
                                    sl_diff = sl_potential_opens - sl_opens
                                    sl_pct_diff = (sl_diff / sl_potential_opens) * 100 if sl_potential_opens > 0 else 0
                                    
                                    new_sl_data.append({
                                        "Subject Line": sl,
                                        "Sends": f"{int(sl_sends):,}",
                                        "Actual Opens": f"{int(sl_opens):,}",
                                        "Actual Rate (%)": f"{sl_actual_rate:.2f}",
                                        "Potential Opens": f"{int(sl_potential_opens):,}",
                                        "Best Rate (%)": f"{best_initial_ctr:.2f}",
                                        "Difference": f"{int(sl_diff):,}",
                                        "Performance": f"{-sl_pct_diff:.2f}%" if sl_pct_diff < 0 else f"-{sl_pct_diff:.2f}%"
                                    })
                            
                            if new_sl_data:
                                new_sl_df = pd.DataFrame(new_sl_data)
                                st.dataframe(new_sl_df)
                            else:
                                st.info("No data available for new subject lines.")
                        else:
                            st.info("No new subject lines were introduced in this simulation.")
                    else:
                        st.warning("Could not identify the best initial subject line.")
                else:
                    st.warning("Insufficient data to calculate metrics.")
        
        # Visualizations Tab
        with tab4:
            st.subheader("Distribution and CTR Visualizations")

            if run_monte_carlo and aggregate_results:
                st.info(f"Showing aggregate results from {n_simulations} simulations with {confidence_level}% confidence intervals")
                
                # Distribution over time - Monte Carlo version
                st.subheader("Average Subject Line Distribution Over Time")
                
                # Collect distribution data across all simulations for each week and subject line
                max_weeks = max(len(df) for df in all_simulation_results)
                
                # Plot average distributions from aggregate_results
                fig, ax = plt.subplots(figsize=(10, 6))
                weeks_range = list(range(1, max_weeks + 1))
                
                for sl in used_sls:
                    if sl in aggregate_results['weekly_distributions']:
                        # Collect mean values for each week
                        values = []
                        for week in range(1, max_weeks + 1):
                            if week in aggregate_results['weekly_distributions'][sl]:
                                values.append(aggregate_results['weekly_distributions'][sl][week]['mean'] * 100)  # Convert to percentage
                            else:
                                values.append(0)
                        
                        # Only plot if subject line has non-zero distribution
                        if any(v > 0 for v in values):
                            ax.plot(weeks_range[:len(values)], values, marker='o', label=sl)
                
                ax.set_xlabel('Week')
                ax.set_ylabel('Distribution (%)')
                ax.set_title('Average Subject Line Distribution Over Time (Across All Simulations)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
                # CTR over time - Monte Carlo version
                st.subheader("Average Cumulative CTR Over Time with Confidence Intervals")
                
                # Create plot with confidence intervals
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for sl in used_sls:
                    if sl in aggregate_results['cumulative_ctrs']:
                        # Collect mean and CI values for each week
                        mean_values = []
                        ci_lows = []
                        ci_highs = []
                        valid_weeks = []
                        
                        for week in range(1, max_weeks + 1):
                            if week in aggregate_results['cumulative_ctrs'][sl]:
                                stats = aggregate_results['cumulative_ctrs'][sl][week]
                                if stats['mean'] > 0:  # Only include weeks with data
                                    mean_values.append(stats['mean'])
                                    ci_lows.append(stats['ci_lower'])
                                    ci_highs.append(stats['ci_upper'])
                                    valid_weeks.append(week)
                        
                        # Only plot if we have data
                        if mean_values:
                            # Plot the mean line
                            ax.plot(valid_weeks, mean_values, marker='o', label=sl)
                            
                            # Add confidence interval
                            ax.fill_between(
                                valid_weeks,
                                ci_lows,
                                ci_highs,
                                alpha=0.2
                            )
                
                ax.set_xlabel('Week')
                ax.set_ylabel('CTR (%)')
                ax.set_title(f'Average Cumulative CTR Over Time with {confidence_level}% Confidence Intervals')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
                # Heat map of final distribution across simulations
                st.subheader("Final Distribution Heatmap")
                st.caption("Shows the distribution that was used in the final week of each simulation (not the prediction for the next week)")
                
                # Collect final distribution data from all simulation DataFrames
                heatmap_data = []
                
                for i, sim_df in enumerate(all_simulation_results):
                    # Get the last real week (not prediction)
                    last_week = sim_df.iloc[-2] if len(sim_df) > n_weeks else sim_df.iloc[-1]
                    
                    row_data = {'Simulation': i + 1}
                    for sl in used_sls:
                        if f'{sl}_distribution' in last_week:
                            row_data[sl] = last_week[f'{sl}_distribution'] * 100  # Convert to percentage
                        else:
                            row_data[sl] = 0
                    
                    heatmap_data.append(row_data)
                
                if heatmap_data:
                    heatmap_df = pd.DataFrame(heatmap_data)
                    
                    # Only keep columns with some distribution
                    cols_to_keep = ['Simulation'] + [col for col in heatmap_df.columns if col != 'Simulation' and heatmap_df[col].sum() > 0]
                    heatmap_df = heatmap_df[cols_to_keep]
                    
                    # Set index to simulation number
                    heatmap_df.set_index('Simulation', inplace=True)
                    
                    # Create heatmap
                    plt.figure(figsize=(12, max(6, len(heatmap_df) * 0.3)))
                    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt=".1f", cbar_kws={'label': 'Distribution (%)'})
                    plt.title('Final Distribution Across Simulations')
                    plt.tight_layout()
                    
                    st.pyplot(plt)
            
            else:
                # Original single-simulation visualizations
                # Distribution over time
                st.subheader("Subject Line Distribution Over Time")
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for sl, values in weekly_dists.items():
                    # Only plot if the subject line has some non-zero distribution
                    if any(v > 0 for v in values):
                        ax.plot(result_df['week'], values, marker='o', label=sl)
                
                ax.set_xlabel('Week')
                ax.set_ylabel('Distribution')
                ax.set_title('Subject Line Distribution Over Time')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
                # Weekly CTR
                st.subheader("Weekly CTR")
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for sl in weekly_ctr_df.columns:
                    ax.plot(weekly_ctr_df.index, weekly_ctr_df[sl], marker='o', label=sl)
                
                ax.set_xlabel('Week')
                ax.set_ylabel('CTR (%)')
                ax.set_title('Weekly CTR by Subject Line')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
                
                # Cumulative CTR
                st.subheader("Cumulative CTR")
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                for sl in cumulative_ctr_df.columns:
                    ax.plot(cumulative_ctr_df.index, cumulative_ctr_df[sl], marker='o', label=sl)
                
                ax.set_xlabel('Week')
                ax.set_ylabel('CTR (%)')
                ax.set_title('Cumulative CTR by Subject Line')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                st.pyplot(fig)
            
            # For both single and Monte Carlo: Epsilon & CTR Relationship
            if run_monte_carlo and aggregate_results:
                st.subheader("Parameter Sensitivity Analysis")
                
                # If we have aggregate results, show epsilon vs CTR
                st.info("Run multiple simulations with different epsilon values to enable this analysis.")
            else:
                st.subheader("New Subject Line Performance")
                
                # Show relationship between new and initial subject lines
                # Identify initial and new subject lines
                initial_sls = []
                for sl in used_sls:
                    for idx, row in result_df.iterrows():
                        if row['week'] == 1 and 'active_sls' in row and sl in row['active_sls'].split(','):
                            initial_sls.append(sl)
                            break
                
                new_sls = [sl for sl in used_sls if sl not in initial_sls]
                
                if new_sls:
                    # Get final CTRs
                    last_week_data = result_df.iloc[-2] if len(result_df) > n_weeks else result_df.iloc[-1]
                    
                    final_ctrs = {}
                    for sl in used_sls:
                        if f'{sl}_cumulative_ctr' in last_week_data:
                            final_ctrs[sl] = last_week_data[f'{sl}_cumulative_ctr']
                    
                    # Create plot comparing initial vs new subject lines
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    x_labels = []
                    ctr_values = []
                    colors = []
                    
                    for sl in sorted(final_ctrs.keys(), key=lambda x: final_ctrs[x], reverse=True):
                        if f'{sl}_cumulative_ctr' in last_week_data and last_week_data[f'{sl}_cumulative_ctr'] > 0:
                            x_labels.append(sl)
                            ctr_values.append(final_ctrs[sl])
                            colors.append('blue' if sl in initial_sls else 'orange')
                    
                    bars = ax.bar(x_labels, ctr_values, color=colors)
                    
                    ax.set_xlabel('Subject Line')
                    ax.set_ylabel('Final CTR (%)')
                    ax.set_title('Final CTR by Subject Line (Blue = Initial, Orange = New)')
                    ax.grid(True, axis='y', alpha=0.3)
                    plt.xticks(rotation=45)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.annotate(f'{height:.2f}%',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom')
                    
                    st.pyplot(fig)
                else:
                    st.info("No new subject lines were introduced during the simulation.")
        
        # Raw Data Tab
        with tab5:
            st.subheader("Raw Data")
            
            data_tabs = st.tabs(["Distribution", "Status", "Weekly Sends", "Weekly Opens", 
                               "Cumulative Sends", "Cumulative Opens", "Weekly CTR", 
                               "Cumulative CTR", "Campaign Metrics", "Complete Data"])
            
            with data_tabs[0]:
                st.dataframe(pd.DataFrame(weekly_dists).T)
            
            with data_tabs[1]:
                st.dataframe(pd.DataFrame(index=result_df['week']).applymap(lambda x: result_df[f'{x}_status']))
            
            with data_tabs[2]:
                st.dataframe(pd.DataFrame(index=result_df['week']).applymap(lambda x: weekly_ctr_df[x].sum()))
            
            with data_tabs[3]:
                st.dataframe(pd.DataFrame(index=result_df['week']).applymap(lambda x: result_df[f'{x}_opens']))
            
            with data_tabs[4]:
                st.dataframe(pd.DataFrame(index=result_df['week']).applymap(lambda x: result_df[f'{x}_cumulative_sends']))
            
            with data_tabs[5]:
                st.dataframe(pd.DataFrame(index=result_df['week']).applymap(lambda x: result_df[f'{x}_cumulative_opens']))
            
            with data_tabs[6]:
                st.dataframe(weekly_ctr_df)
            
            with data_tabs[7]:
                st.dataframe(cumulative_ctr_df)
                
            with data_tabs[8]:
                # Select the campaign metrics columns
                campaign_metrics_cols = ['week', 'overall_weekly_ctr', 'initial_sl_weekly_ctr', 'new_sl_weekly_ctr',
                                       'cumulative_campaign_ctr', 'cumulative_initial_sl_ctr', 
                                       'counterfactual_ctr', 'mab_benefit', 'num_new_active_sls']
                campaign_metrics_df = result_df[campaign_metrics_cols].copy() if all(col in result_df.columns for col in campaign_metrics_cols) else pd.DataFrame()
                
                if not campaign_metrics_df.empty:
                    st.dataframe(campaign_metrics_df)
                else:
                    st.info("Campaign metrics data not available.")
            
            with data_tabs[9]:
                st.dataframe(result_df)
    
    st.success(f"Simulation completed for up to {n_weeks} weeks with {n_initial_subjectlines} initial subject lines and a maximum of {max_subjectlines} subject lines")

else:
    st.info("Set the simulation parameters and click 'Run Simulation' to start.") 