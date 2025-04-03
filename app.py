import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import generate_weekly_sends
import time

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

# Open Rate Settings
st.sidebar.header("Subject Line Open Rates")
use_custom_rates = st.sidebar.checkbox("Customize Open Rates", value=False)

# Default presets for open rates
default_min_rates = {
    "SL_1": 0.15, "SL_2": 0.12, "SL_3": 0.18, "SL_4": 0.14, "SL_5": 0.16,
    "SL_6": 0.13, "SL_7": 0.17, "SL_8": 0.11, "SL_9": 0.19, "SL_10": 0.15,
    "SL_11": 0.14, "SL_12": 0.16, "SL_13": 0.13, "SL_14": 0.18, "SL_15": 0.12,
    "SL_16": 0.17, "SL_17": 0.15, "SL_18": 0.19, "SL_19": 0.14, "SL_20": 0.16
}

default_max_rates = {
    "SL_1": 0.22, "SL_2": 0.19, "SL_3": 0.25, "SL_4": 0.21, "SL_5": 0.23,
    "SL_6": 0.20, "SL_7": 0.24, "SL_8": 0.18, "SL_9": 0.26, "SL_10": 0.22,
    "SL_11": 0.21, "SL_12": 0.23, "SL_13": 0.20, "SL_14": 0.25, "SL_15": 0.19,
    "SL_16": 0.24, "SL_17": 0.22, "SL_18": 0.26, "SL_19": 0.21, "SL_20": 0.23
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
                min_rate = st.number_input(f"{sl_name} Min", min_value=0.01, max_value=0.50, value=default_min_rates[sl_name], step=0.01, format="%.2f")
            with cols[1]:
                max_rate = st.number_input(f"{sl_name} Max", min_value=min_rate, max_value=0.60, value=max(min_rate+0.05, default_max_rates[sl_name]), step=0.01, format="%.2f")
            custom_min_rates[sl_name] = min_rate
            custom_max_rates[sl_name] = max_rate
    
    with st.sidebar.expander("Additional Subject Lines"):
        for i in range(n_initial_subjectlines + 1, max_subjectlines + 1):
            sl_name = f"SL_{i}"
            cols = st.columns(2)
            with cols[0]:
                min_rate = st.number_input(f"{sl_name} Min", min_value=0.01, max_value=0.50, value=default_min_rates[sl_name], step=0.01, format="%.2f")
            with cols[1]:
                max_rate = st.number_input(f"{sl_name} Max", min_value=min_rate, max_value=0.60, value=max(min_rate+0.05, default_max_rates[sl_name]), step=0.01, format="%.2f")
            custom_min_rates[sl_name] = min_rate
            custom_max_rates[sl_name] = max_rate

# Run simulation button
if st.sidebar.button("Run Simulation"):
    with st.spinner("Running simulation..."):
        # Create containers to capture the print outputs
        simulation_output = st.empty()
        
        # Redirect stdout to capture print statements
        import io
        import contextlib
        import sys
        
        # Create a StringIO object to capture output
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # Run the simulation with custom open rates if specified
            result_df = generate_weekly_sends(
                n_weeks=n_weeks, 
                n_subjectlines=n_initial_subjectlines, 
                max_subjectlines=max_subjectlines,
                epsilon=epsilon,
                new_sl_weeks=new_sl_weeks,
                custom_min_rates=custom_min_rates if use_custom_rates else None,
                custom_max_rates=custom_max_rates if use_custom_rates else None
            )
        
        # Get the captured output
        output_text = f.getvalue()
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Simulation Log", "Summary", "Metrics", "Visualizations", "Raw Data"])
        
        with tab1:
            st.subheader("Simulation Log")
            st.text(output_text)
        
        # Get all possible subject line names
        all_sl_names = [f"SL_{i}" for i in range(1, max_subjectlines + 1)]
        
        # Find which subject lines were actually used in the simulation
        used_sl_cols = [col for col in result_df.columns if any(col.startswith(f"{sl}_") for sl in all_sl_names)]
        used_sls = sorted(list(set([col.split('_')[0] + '_' + col.split('_')[1] for col in used_sl_cols])))
        
        # Create a function to get active subject lines for each week
        def get_active_sls_for_week(week_row):
            return week_row['active_sls'].split(',') if 'active_sls' in week_row else []
        
        # Extract data for each visualization from the result DataFrame
        
        # Distribution data - only include active subject lines for each week
        dist_data = {}
        for sl in used_sls:
            if f'{sl}_distribution' in result_df.columns:
                dist_data[sl] = result_df[f'{sl}_distribution'] * 100
        
        dist_df = pd.DataFrame(dist_data)
        dist_df.index = [f"Week {int(w)}" for w in result_df['week']]
        
        # Weekly sends data
        sends_data = {}
        for sl in used_sls:
            if f'{sl}_sends' in result_df.columns:
                sends_data[sl] = result_df[f'{sl}_sends']
        
        weekly_sends_df = pd.DataFrame(sends_data)
        weekly_sends_df.index = [f"Week {int(w)}" for w in result_df['week']]
        
        # Weekly opens data
        opens_data = {}
        for sl in used_sls:
            if f'{sl}_opens' in result_df.columns:
                opens_data[sl] = result_df[f'{sl}_opens']
        
        weekly_opens_df = pd.DataFrame(opens_data)
        weekly_opens_df.index = [f"Week {int(w)}" for w in result_df['week']]
        
        # Cumulative sends data
        cum_sends_data = {}
        for sl in used_sls:
            if f'{sl}_cumulative_sends' in result_df.columns:
                cum_sends_data[sl] = result_df[f'{sl}_cumulative_sends']
        
        cumulative_sends_df = pd.DataFrame(cum_sends_data)
        cumulative_sends_df.index = [f"Week {int(w)}" for w in result_df['week']]
        
        # Cumulative opens data
        cum_opens_data = {}
        for sl in used_sls:
            if f'{sl}_cumulative_opens' in result_df.columns:
                cum_opens_data[sl] = result_df[f'{sl}_cumulative_opens']
        
        cumulative_opens_df = pd.DataFrame(cum_opens_data)
        cumulative_opens_df.index = [f"Week {int(w)}" for w in result_df['week']]
        
        # Weekly CTR data
        weekly_ctr_data = {}
        for sl in used_sls:
            if f'{sl}_weekly_ctr' in result_df.columns:
                weekly_ctr_data[sl] = result_df[f'{sl}_weekly_ctr']
        
        weekly_ctr_df = pd.DataFrame(weekly_ctr_data)
        weekly_ctr_df.index = [f"Week {int(w)}" for w in result_df['week']]
        
        # Cumulative CTR data
        cum_ctr_data = {}
        for sl in used_sls:
            if f'{sl}_cumulative_ctr' in result_df.columns:
                cum_ctr_data[sl] = result_df[f'{sl}_cumulative_ctr']
        
        cumulative_ctr_df = pd.DataFrame(cum_ctr_data)
        cumulative_ctr_df.index = [f"Week {int(w)}" for w in result_df['week']]
        
        # Status data
        status_data = {}
        for sl in used_sls:
            if f'{sl}_status' in result_df.columns:
                status_data[sl] = result_df[f'{sl}_status']
        
        status_df = pd.DataFrame(status_data)
        status_df.index = [f"Week {int(w)}" for w in result_df['week']]
        
        # Summary Tab
        with tab2:
            st.subheader("Simulation Summary")
            
            # Find if simulation ended early with a winner
            simulation_ended_early = len(result_df) <= n_weeks  # No prediction row means it ended early
            winner_found = False
            winner_sl = None
            winner_week = None
            
            for idx, row in result_df.iterrows():
                for sl in used_sls:
                    if f'{sl}_distribution' in row and row[f'{sl}_distribution'] == 1.0:
                        winner_found = True
                        winner_sl = sl
                        winner_week = int(row['week'])
                        break
                if winner_found:
                    break
            
            # Get last week data (excluding the prediction row if it exists)
            last_week_data = result_df.iloc[-2] if len(result_df) > n_weeks else result_df.iloc[-1]
            
            # Overall metrics
            cols = st.columns(4)
            
            # Total active subject lines used
            used_sl_count = len(set(sl for week_active_sls in [row['active_sls'].split(',') for _, row in result_df.iterrows() if 'active_sls' in row] for sl in week_active_sls))
            
            # Overall metrics
            total_sends = sum(last_week_data[[f'{sl}_cumulative_sends' for sl in used_sls if f'{sl}_cumulative_sends' in last_week_data]])
            total_opens = sum(last_week_data[[f'{sl}_cumulative_opens' for sl in used_sls if f'{sl}_cumulative_opens' in last_week_data]])
            overall_ctr = total_opens / total_sends * 100 if total_sends > 0 else 0
            
            with cols[0]:
                st.metric("Total Sends", f"{int(total_sends):,}")
            with cols[1]:
                st.metric("Total Opens", f"{int(total_opens):,}")
            with cols[2]:
                st.metric("Overall CTR", f"{overall_ctr:.2f}%")
            with cols[3]:
                st.metric("Subject Lines Used", f"{used_sl_count}")
            
            # Winner information if found
            if winner_found:
                st.success(f"🏆 Winner found! {winner_sl} reached 100% distribution in Week {winner_week}.")
            
            # Subject line status tracking
            st.subheader("Subject Line Introduction and Status")
            
            # Create a table showing when each subject line was introduced and its status
            intro_data = []
            for sl in used_sls:
                # Find the first week this subject line was active
                intro_week = None
                final_status = "inactive"
                final_dist = 0.0
                cumulative_sends = 0
                cumulative_opens = 0
                cumulative_ctr = 0.0
                
                for idx, row in result_df.iterrows():
                    active_sls = row['active_sls'].split(',') if 'active_sls' in row else []
                    
                    if sl in active_sls and intro_week is None:
                        intro_week = int(row['week'])
                    
                    if f'{sl}_status' in row:
                        final_status = row[f'{sl}_status']
                    
                    if f'{sl}_distribution' in row:
                        final_dist = row[f'{sl}_distribution'] * 100
                    
                    if f'{sl}_cumulative_sends' in row:
                        cumulative_sends = row[f'{sl}_cumulative_sends']
                    
                    if f'{sl}_cumulative_opens' in row:
                        cumulative_opens = row[f'{sl}_cumulative_opens']
                
                if cumulative_sends > 0:
                    cumulative_ctr = cumulative_opens / cumulative_sends * 100
                
                if intro_week is not None:
                    intro_data.append({
                        'Subject Line': sl,
                        'Introduced Week': intro_week,
                        'Final Status': final_status,
                        'Final Distribution (%)': final_dist,
                        'Total Sends': cumulative_sends,
                        'Total Opens': cumulative_opens,
                        'CTR (%)': cumulative_ctr
                    })
            
            intro_df = pd.DataFrame(intro_data)
            if not intro_df.empty:
                st.dataframe(intro_df.sort_values(['Introduced Week', 'Subject Line']))
            
            # Final distribution
            st.subheader("Final Subject Line Distribution")
            
            # Get final distributions
            final_week = result_df.iloc[-1]
            final_dist_data = {}
            for sl in used_sls:
                if f'{sl}_distribution' in final_week:
                    if final_week[f'{sl}_status'] != 'inactive':
                        final_dist_data[sl] = final_week[f'{sl}_distribution'] * 100
            
            final_dist_df = pd.DataFrame([final_dist_data])
            final_dist_df.index = ["Final Distribution (%)"]
            st.dataframe(final_dist_df.T.sort_values("Final Distribution (%)", ascending=False))
            
            # Week by week summary
            st.subheader("Week by Week Summary")
            summary_data = []
            for idx, row in result_df.iloc[:-1].iterrows() if len(result_df) > n_weeks else result_df.iterrows():
                week_num = int(row['week'])
                total_week_sends = row['total_sends']
                
                # Active subject lines for this week
                active_sls_week = row['active_sls'].split(',') if 'active_sls' in row else []
                
                # Count new and standard subject lines
                new_sls = [sl for sl in active_sls_week if f'{sl}_status' in row and row[f'{sl}_status'] == 'new']
                standard_sls = [sl for sl in active_sls_week if f'{sl}_status' in row and row[f'{sl}_status'] == 'standard']
                
                # Total opens and CTR for this week
                total_week_opens = sum([row[f'{sl}_opens'] for sl in active_sls_week if f'{sl}_opens' in row])
                week_ctr = total_week_opens / total_week_sends * 100 if total_week_sends > 0 else 0
                
                # Find best performing subject line
                best_sl = None
                best_sl_ctr = 0
                for sl in active_sls_week:
                    if f'{sl}_weekly_ctr' in row and row[f'{sl}_weekly_ctr'] > best_sl_ctr:
                        best_sl_ctr = row[f'{sl}_weekly_ctr']
                        best_sl = sl
                
                summary_data.append({
                    'Week': week_num,
                    'Total Sends': total_week_sends,
                    'Total Opens': total_week_opens,
                    'CTR (%)': week_ctr,
                    'Active SLs': len(active_sls_week),
                    'New SLs': len(new_sls),
                    'Standard SLs': len(standard_sls),
                    'Best Subject Line': best_sl,
                    'Best SL CTR (%)': best_sl_ctr
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
        
        # Metrics Tab
        with tab3:
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
                
                **Exploration Cost Calculation:**
                1. For each new subject line, we calculate:
                   - Actual Opens: The actual number of opens received
                   - Potential Opens: The opens we would have received if we used the best initial subject line instead
                   - The difference (if positive) is the exploration cost
                
                **Weekly Performance Delta:**
                - For each week where new subject lines are introduced:
                   - We compare the CTR of each new subject line against the best-performing initial subject line that week
                   - Positive delta means the new subject line outperformed the best initial one
                
                **Parameter Tuning Metrics:**
                - **Exploration Efficiency**: Ratio of actual opens from new subject lines to potential opens (higher is better)
                - **Distribution Balance**: How evenly distributed the final allocation is (closer to 1 = more balanced)
                - **Average Weekly CTR**: Simple average of all weekly CTRs across all subject lines
                """)
            
            # Get last week data (excluding the prediction row if it exists)
            last_week_data = result_df.iloc[-2] if len(result_df) > n_weeks else result_df.iloc[-1]
            
            # 1. Identify initial subject lines (first n_initial_subjectlines)
            initial_sls = [f"SL_{i}" for i in range(1, n_initial_subjectlines + 1)]
            
            # Check if they exist in the data
            initial_sls = [sl for sl in initial_sls if f'{sl}_cumulative_ctr' in last_week_data]
            
            # 2. Identify new subject lines (introduced after week 1)
            new_sls = []
            for sl in used_sls:
                for idx, row in result_df.iterrows():
                    if row['week'] > 1 and 'active_sls' in row and sl in row['active_sls']:
                        if sl not in new_sls and sl not in initial_sls:
                            new_sls.append(sl)
            
            # 3. Calculate metrics for initial subject lines
            if initial_sls:
                # Find best performing initial subject line
                initial_sl_ctrs = {sl: last_week_data[f'{sl}_cumulative_ctr'] 
                                 for sl in initial_sls 
                                 if f'{sl}_cumulative_ctr' in last_week_data}
                
                if initial_sl_ctrs:
                    best_initial_sl = max(initial_sl_ctrs.items(), key=lambda x: x[1])[0]
                    best_initial_ctr = initial_sl_ctrs[best_initial_sl]
                    
                    st.subheader("Best Initial Subject Line")
                    with st.expander("ℹ️ How is the best initial subject line determined?"):
                        st.markdown(f"""
                        The best initial subject line is determined by looking at all subject lines introduced in Week 1 
                        and selecting the one with the highest cumulative CTR at the end of the simulation.
                        
                        In this simulation, **{best_initial_sl}** had the highest cumulative CTR of **{best_initial_ctr:.2f}%** among initial subject lines.
                        
                        This subject line serves as the baseline for measuring the opportunity cost of exploration.
                        """)
                    
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Best Initial SL", best_initial_sl)
                    with cols[1]:
                        st.metric("CTR", f"{best_initial_ctr:.2f}%")
                    with cols[2]:
                        total_initial_opens = sum([last_week_data[f'{sl}_cumulative_opens'] 
                                                for sl in initial_sls 
                                                if f'{sl}_cumulative_opens' in last_week_data])
                        st.metric("Total Opens", f"{int(total_initial_opens):,}")
                    
                    # 4. Calculate exploration cost
                    exploration_cost = 0
                    potential_opens = 0
                    actual_opens = 0
                    
                    # For each new subject line, calculate potential opens if best initial SL was used
                    for sl in new_sls:
                        if f'{sl}_cumulative_sends' in last_week_data and f'{sl}_cumulative_opens' in last_week_data:
                            sl_sends = last_week_data[f'{sl}_cumulative_sends']
                            sl_opens = last_week_data[f'{sl}_cumulative_opens']
                            
                            # Potential opens if best initial SL was used instead
                            potential_sl_opens = sl_sends * (best_initial_ctr / 100)
                            
                            exploration_cost += max(0, potential_sl_opens - sl_opens)
                            potential_opens += potential_sl_opens
                            actual_opens += sl_opens
                    
                    st.subheader("Exploration Cost")
                    with st.expander("ℹ️ How is exploration cost calculated?"):
                        st.markdown(f"""
                        **Exploration cost** represents the number of opens sacrificed by testing new subject lines instead of using the best initial subject line.
                        
                        **Formula:**
                        ```
                        For each new subject line:
                            Potential Opens = Sends to New SL × (Best Initial SL CTR / 100)
                            Exploration Cost = max(0, Potential Opens - Actual Opens)
                        ```
                        
                        **In this simulation:**
                        - New subject lines received **{int(actual_opens):,}** actual opens
                        - If we had sent all those emails using the best initial subject line ({best_initial_sl}), 
                          we would have received approximately **{int(potential_opens):,}** opens
                        - This results in **{int(exploration_cost):,}** sacrificed opens, or **{exploration_cost/total_opens*100:.2f}%** of total opens
                        
                        Note: If exploration cost is negative (new SLs outperform the best initial SL), it's set to zero.
                        """)
                    
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Sacrificed Opens", f"{int(exploration_cost):,}")
                    with cols[1]:
                        total_opens = sum([last_week_data[f'{sl}_cumulative_opens'] 
                                        for sl in used_sls 
                                        if f'{sl}_cumulative_opens' in last_week_data])
                        st.metric("Sacrifice Percentage", f"{exploration_cost/total_opens*100:.2f}%" if total_opens > 0 else "0.00%")
                    with cols[2]:
                        st.metric("Potential Opens (New SLs)", f"{int(potential_opens):,}")
                    with cols[3]:
                        st.metric("Actual Opens (New SLs)", f"{int(actual_opens):,}")
                    
                    # 5. Calculate weekly performance delta for new subject lines
                    st.subheader("Weekly Performance Delta for New Subject Lines")
                    with st.expander("ℹ️ How is performance delta calculated?"):
                        st.markdown("""
                        **Performance delta** measures how new subject lines perform relative to the best initial subject line on a weekly basis.
                        
                        **Formula:**
                        ```
                        For each week where new SLs are introduced:
                            Delta = New SL CTR - Best Initial SL CTR for that week
                        ```
                        
                        **Interpretation:**
                        - **Positive delta**: The new subject line outperformed the best initial subject line that week
                        - **Negative delta**: The new subject line underperformed compared to the best initial subject line
                        
                        The chart shows the average delta across all new subject lines introduced each week.
                        
                        This metric helps evaluate whether exploration of new subject lines is yielding better options
                        than sticking with the initial set.
                        """)
                    
                    delta_data = []
                    for _, row in result_df.iterrows():
                        week_num = int(row['week'])
                        if week_num == 1:  # Skip week 1
                            continue
                            
                        # Get active subject lines for this week
                        active_sls = row['active_sls'].split(',') if 'active_sls' in row else []
                        
                        # Find new subject lines introduced this week
                        week_new_sls = []
                        for sl in active_sls:
                            if sl not in initial_sls:
                                # Check if this was the first week for this subject line
                                is_first_week = True
                                for prev_idx, prev_row in result_df.iterrows():
                                    if prev_row['week'] < week_num and 'active_sls' in prev_row and sl in prev_row['active_sls'].split(','):
                                        is_first_week = False
                                        break
                                
                                if is_first_week:
                                    week_new_sls.append(sl)
                        
                        if week_new_sls:
                            # Calculate performance for best initial SL this week
                            best_initial_weekly_ctr = 0
                            best_initial_weekly_sl = None
                            
                            for sl in initial_sls:
                                if sl in active_sls and f'{sl}_weekly_ctr' in row:
                                    if row[f'{sl}_weekly_ctr'] > best_initial_weekly_ctr:
                                        best_initial_weekly_ctr = row[f'{sl}_weekly_ctr']
                                        best_initial_weekly_sl = sl
                            
                            # Calculate delta for each new SL
                            for new_sl in week_new_sls:
                                if f'{new_sl}_weekly_ctr' in row and best_initial_weekly_sl:
                                    new_sl_ctr = row[f'{new_sl}_weekly_ctr']
                                    delta = new_sl_ctr - best_initial_weekly_ctr
                                    
                                    delta_data.append({
                                        'Week': week_num,
                                        'New Subject Line': new_sl,
                                        'New SL CTR (%)': new_sl_ctr,
                                        'Best Initial SL': best_initial_weekly_sl,
                                        'Best Initial CTR (%)': best_initial_weekly_ctr,
                                        'Delta (%)': delta
                                    })
                    
                    # Display delta data
                    if delta_data:
                        delta_df = pd.DataFrame(delta_data)
                        st.dataframe(delta_df)
                        
                        # Visualize delta
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Group by week and calculate average delta
                        weekly_avg_delta = delta_df.groupby('Week')['Delta (%)'].mean().reset_index()
                        
                        ax.bar(weekly_avg_delta['Week'], weekly_avg_delta['Delta (%)'])
                        ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                        plt.title('Average CTR Delta (New SLs vs Best Initial SL)')
                        plt.xlabel('Week')
                        plt.ylabel('Delta (%)')
                        plt.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    else:
                        st.info("No new subject lines were introduced after week 1.")
                else:
                    st.warning("Could not calculate metrics - not enough data for initial subject lines.")
            else:
                st.warning("Could not identify initial subject lines in the data.")
                
            # 6. Add summary metrics for parameter tuning
            st.subheader("Parameter Tuning Metrics")
            with st.expander("ℹ️ How to interpret parameter tuning metrics?"):
                st.markdown("""
                **Parameter tuning metrics** help you optimize the MAB simulation settings:
                
                1. **Simulation Duration**: Number of weeks the simulation ran for, which may be less than the configured maximum if a winner was found early
                
                2. **Exploration Efficiency**: Calculated as (Actual Opens from New SLs) ÷ (Potential Opens from New SLs) × 100
                   - A value close to 100% means new subject lines performed nearly as well as the best initial subject line
                   - Higher values are better, indicating effective exploration
                
                3. **Avg Weekly CTR**: The average CTR across all weeks and all active subject lines
                   - Useful for comparing overall performance across simulation runs
                
                4. **Distribution Balance**: Measures how evenly the final distribution is spread among subject lines
                   - Calculated as 1 - (Standard Deviation ÷ Mean) for final distribution values
                   - A value close to 1 indicates an even distribution
                   - A value close to 0 indicates allocation concentrated on few subject lines
                
                **How to use these metrics:**
                Run multiple simulations with different parameter combinations (ε, New SL Weeks, Max SLs) to find the optimal configuration 
                that balances exploration and exploitation for your use case.
                """)
                
            total_weeks = max(result_df['week']) if not result_df.empty else 0
            
            # Calculate exploration efficiency
            if 'potential_opens' in locals() and potential_opens > 0:
                exploration_efficiency = actual_opens / potential_opens * 100
            else:
                exploration_efficiency = 0
                
            # Calculate average weekly CTR
            weekly_ctr_avg = weekly_ctr_df.mean().mean() if not weekly_ctr_df.empty else 0
            
            # Calculate final allocation concentration (how evenly distributed)
            final_dist = [last_week_data[f'{sl}_distribution'] for sl in used_sls if f'{sl}_distribution' in last_week_data and last_week_data[f'{sl}_distribution'] > 0]
            if final_dist:
                final_dist_concentration = 1 - (np.std(final_dist) / np.mean(final_dist) if np.mean(final_dist) > 0 else 0)
            else:
                final_dist_concentration = 0
                
            # Display metrics
            cols = st.columns(4)
            with cols[0]:
                st.metric("Simulation Duration", f"{total_weeks} weeks")
            with cols[1]:
                st.metric("Exploration Efficiency", f"{exploration_efficiency:.2f}%")
            with cols[2]:
                st.metric("Avg Weekly CTR", f"{weekly_ctr_avg:.2f}%")
            with cols[3]:
                st.metric("Distribution Balance", f"{final_dist_concentration:.2f}")
                
            # Parameter combination performance
            st.info(f"Parameter Configuration: ε={epsilon}, New SL Weeks={new_sl_weeks}, Max SLs={max_subjectlines}")
        
        # Visualizations Tab
        with tab4:
            st.subheader("Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Distribution over time
                st.subheader("Distribution Over Time")
                # Exclude the last row (prediction) if it exists
                plot_dist_df = dist_df.iloc[:-1] if len(result_df) > n_weeks else dist_df
                
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_dist_df.plot(ax=ax)
                plt.title("Subject Line Distribution Over Time")
                plt.xlabel("Week")
                plt.ylabel("Distribution (%)")
                plt.legend(title="Subject Line")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Subject Line Status
                st.subheader("Subject Line Status Over Time")
                
                # Create a heatmap of status
                status_values = {'inactive': 0, 'new': 1, 'standard': 2}
                status_numeric = status_df.applymap(lambda x: status_values.get(x, 0))
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(status_numeric.T, cmap=['white', 'blue', 'green'], 
                           cbar_kws={'ticks': [0.5, 1.5, 2.5], 'label': 'Status'},
                           linewidths=0.5)
                ax.set_yticks(np.arange(len(status_numeric.columns)) + 0.5)
                ax.set_yticklabels(status_numeric.columns)
                plt.colorbar(ax.collections[0], ticks=[0.5, 1.5, 2.5], 
                            label='Status')
                ax.collections[0].colorbar.set_ticklabels(['Inactive', 'New', 'Standard'])
                plt.title("Subject Line Status by Week")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Cumulative CTR
                st.subheader("Cumulative CTR Over Time")
                
                # Only plot for subject lines with data
                valid_cols = [col for col in cumulative_ctr_df.columns 
                             if cumulative_ctr_df[col].max() > 0]
                plot_ctr_df = cumulative_ctr_df[valid_cols]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_ctr_df.iloc[:-1].plot(ax=ax) if len(result_df) > n_weeks else plot_ctr_df.plot(ax=ax)
                plt.title("Cumulative CTR by Subject Line")
                plt.xlabel("Week")
                plt.ylabel("CTR (%)")
                plt.legend(title="Subject Line")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with viz_col2:
                # Weekly sends
                st.subheader("Weekly Sends by Subject Line")
                
                # Only plot for subject lines with data
                valid_cols = [col for col in weekly_sends_df.columns 
                             if weekly_sends_df[col].max() > 0]
                plot_sends_df = weekly_sends_df[valid_cols]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                (plot_sends_df.iloc[:-1].plot.bar(ax=ax, stacked=True) 
                 if len(result_df) > n_weeks else plot_sends_df.plot.bar(ax=ax, stacked=True))
                plt.title("Weekly Sends by Subject Line")
                plt.xlabel("Week")
                plt.ylabel("Number of Sends")
                plt.legend(title="Subject Line")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Weekly CTR
                st.subheader("Weekly CTR by Subject Line")
                
                # Only plot for subject lines with data
                valid_cols = [col for col in weekly_ctr_df.columns 
                             if weekly_ctr_df[col].max() > 0]
                plot_weekly_ctr_df = weekly_ctr_df[valid_cols]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                (plot_weekly_ctr_df.iloc[:-1].plot(ax=ax, marker='o') 
                 if len(result_df) > n_weeks else plot_weekly_ctr_df.plot(ax=ax, marker='o'))
                plt.title("Weekly CTR by Subject Line")
                plt.xlabel("Week")
                plt.ylabel("CTR (%)")
                plt.legend(title="Subject Line")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Additional visualizations spanning the full width
            st.subheader("Final Performance Comparison")
            
            # Get active subject lines in the last actual week (not prediction)
            last_week = result_df.iloc[-2] if len(result_df) > n_weeks else result_df.iloc[-1]
            last_week_active_sls = last_week['active_sls'].split(',') if 'active_sls' in last_week else []
            
            # Only include active subject lines
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(last_week_active_sls))
            width = 0.35
            
            last_week_sends = [last_week[f'{sl}_cumulative_sends'] if f'{sl}_cumulative_sends' in last_week else 0 
                              for sl in last_week_active_sls]
            last_week_opens = [last_week[f'{sl}_cumulative_opens'] if f'{sl}_cumulative_opens' in last_week else 0 
                              for sl in last_week_active_sls]
            
            ax.bar(x - width/2, last_week_sends, width, label='Sends')
            ax.bar(x + width/2, last_week_opens, width, label='Opens')
            ax.set_xticks(x)
            ax.set_xticklabels(last_week_active_sls)
            ax.legend()
            plt.title("Final Send and Open Counts by Subject Line")
            plt.xlabel("Subject Line")
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Evolution of distributions
            st.subheader("Evolution of Subject Line Distribution")
            
            # Only plot for subject lines that were active at some point
            active_sls_ever = []
            for _, row in result_df.iterrows():
                if 'active_sls' in row:
                    active_sls_ever.extend(row['active_sls'].split(','))
            active_sls_ever = list(set(active_sls_ever))
            
            # Create a dataframe with just the active subject lines
            active_dist_data = {}
            for sl in active_sls_ever:
                if f'{sl}_distribution' in result_df.columns:
                    active_dist_data[sl] = result_df[f'{sl}_distribution'] * 100
            
            active_dist_df = pd.DataFrame(active_dist_data)
            active_dist_df.index = [f"Week {int(w)}" for w in result_df['week']]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create a stacked area chart
            if len(result_df) > n_weeks:
                active_dist_df.iloc[:-1].plot.area(ax=ax, stacked=True, alpha=0.7)
            else:
                active_dist_df.plot.area(ax=ax, stacked=True, alpha=0.7)
                
            plt.title("Evolution of Subject Line Distribution")
            plt.xlabel("Week")
            plt.ylabel("Distribution (%)")
            plt.legend(title="Subject Line")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Final distribution pie chart
            st.subheader("Final Distribution Pie Chart")
            
            # Get final distribution for active subject lines
            final_week = result_df.iloc[-1]
            active_sls_final = final_week['active_sls'].split(',') if 'active_sls' in final_week else []
            
            # Filter out zero distributions
            labels = []
            sizes = []
            
            for sl in active_sls_final:
                if f'{sl}_distribution' in final_week and final_week[f'{sl}_distribution'] > 0:
                    labels.append(sl)
                    sizes.append(final_week[f'{sl}_distribution'] * 100)
            
            if sizes:  # Only create pie chart if there are non-zero values
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.title("Final Subject Line Distribution")
                st.pyplot(fig)
            else:
                st.write("No non-zero distribution values to display.")
        
        # Raw Data Tab
        with tab5:
            st.subheader("Raw Data")
            
            data_tabs = st.tabs(["Distribution", "Status", "Weekly Sends", "Weekly Opens", 
                               "Cumulative Sends", "Cumulative Opens", "Weekly CTR", 
                               "Cumulative CTR", "Complete Data"])
            
            with data_tabs[0]:
                st.dataframe(dist_df)
            
            with data_tabs[1]:
                st.dataframe(status_df)
            
            with data_tabs[2]:
                st.dataframe(weekly_sends_df)
            
            with data_tabs[3]:
                st.dataframe(weekly_opens_df)
            
            with data_tabs[4]:
                st.dataframe(cumulative_sends_df)
            
            with data_tabs[5]:
                st.dataframe(cumulative_opens_df)
            
            with data_tabs[6]:
                st.dataframe(weekly_ctr_df)
            
            with data_tabs[7]:
                st.dataframe(cumulative_ctr_df)
            
            with data_tabs[8]:
                st.dataframe(result_df)
    
    st.success(f"Simulation completed for up to {n_weeks} weeks with {n_initial_subjectlines} initial subject lines and a maximum of {max_subjectlines} subject lines")

else:
    st.info("Set the simulation parameters and click 'Run Simulation' to start.") 