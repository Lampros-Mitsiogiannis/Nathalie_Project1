import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Function to load data from the uploaded file
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df['Effective Date'] = pd.to_datetime(df['Effective Date'])
    df['Month-Year'] = df['Effective Date'].dt.strftime('%b %Y')  # e.g., "Jan 2023"
    # This will create a sortable month-year column like '202301' for 'Jan 2023'
    df['Sortable Month-Year'] = df['Effective Date'].dt.strftime('%Y%m')
    # Extract only the numbers from the 'Owner' column
    df['Owner_Number'] = df['Owner'].str.extract('(\d+)')

    country_codes = {
        '11': 'Sonova AG',
        '13': 'Sonova Communications AG',
        "3250": 'RCSE',
        '51': 'Sonova USA',
        '52': 'Sonova Canada',
        '53': 'Unitron',
        '6255': 'Mexicali',
        '74': 'SOCC',
        '84': 'SOCV'
    }

    # Map the country codes to the 'Owner_Number' column to create a new 'Country' column
    df['Country'] = df['Owner_Number'].map(country_codes)
    return df

# Function to create a pivot table
def create_pivot_table(df, status_filter, type_filter):
    # Sort the DataFrame by 'Sortable Month-Year' to ensure months are in the correct order
    df_sorted = df.sort_values('Sortable Month-Year')
    # Filter based on selected status and type
    filtered_df = df[df['Status'].isin(status_filter) & df['Type'].isin(type_filter)]
    # Create a pivot table with 'Type' as rows, 'Month-Year' as columns, and count of 'SDC_ID' as values
    pivot_table = pd.pivot_table(
        filtered_df,
        values='SDC_ID',
        index='Type',
        columns='Month-Year',
        aggfunc='count',
        fill_value=0
    )
    pivot_table = pivot_table.reindex(
        columns=sorted(pivot_table.columns, key=lambda x: pd.to_datetime(x, format='%b %Y')))
    return pivot_table

# Function to get a formatted year range from a DataFrame
def get_year_range(df):
    years = df['Effective Date'].dt.year.unique()
    print(combined_data['Effective Date'].dt.year.unique())
    if len(years) > 1:
        return f"{min(years)}-{max(years)}"
    return str(years[0])

def create_status_pivot(df):
    current_date = pd.to_datetime('today')

    # Ensure the date columns are in datetime format
    df['Created Date'] = pd.to_datetime(df['Created Date'])
    df['A_DUE_DATE'] = pd.to_datetime(df['A_DUE_DATE'])
    df['A_CLOSED_DATE'] = pd.to_datetime(df['A_CLOSED_DATE'])

    df['Sortable Month-Year'] = df['Created Date'].dt.strftime('%Y%m')  # Sortable as 'YYYYMM'

    # Ensure we're sorting the DataFrame in ascending order
    df.sort_values('Sortable Month-Year', inplace=True)


    # Calculate the number of days for INWORKS and CLOSED
    df['Days in INWORKS'] = (current_date - df['Created Date']).dt.days
    df['Days to CLOSE'] = (df['A_CLOSED_DATE'] - df['Created Date']).dt.days

    # Replace negative and NaN values with None to exclude them from analysis
    df['Days to CLOSE'] = df['Days to CLOSE'].apply(lambda x: x if x > 0 else None)

    # Function to categorize the days into intervals for CLOSED status
    def categorize_days_closed(days):
        if days is None:
            return '0'  # Exclude unclosed records
        elif days < 90:
            return '<90 days'
        elif 90 <= days < 180:
            return '90-180 days'
        elif 180 <= days < 360:
            return '180-360 days'
        else:
            return '>360 days'

    # Function to categorize the days into intervals for INWORKS status
    def categorize_days_inworks(days):
        if days is None:
            return '0'  # Exclude unclosed records
        elif days < 90:
            return '<90 days'
        elif 90 <= days < 180:
            return '90-180 days'
        elif 180 <= days < 360:
            return '180-360 days'
        else:
            return '>360 days'

    # Apply the categorization
    df['INWORKS_Category'] = df['Days in INWORKS'].apply(categorize_days_inworks)
    df['CLOSED_Category'] = df['Days to CLOSE'].apply(categorize_days_closed)

    # Create a sortable month-year column for chronological sorting
    df['Sortable Month-Year'] = df['Created Date'].dt.strftime('%Y%m')

    # Extract the month and year from 'Created Date' for display purposes
    df['Month-Year'] = df['Created Date'].dt.strftime('%b %Y')

    # Create pivot tables for INWORKS and CLOSED status with months as columns
    inworks_pivot = df[df['A_PRIMARY_STATUS'] == 'INWORKS'].pivot_table(
        index='INWORKS_Category',
        columns='Sortable Month-Year',
        values='A_PK_ID',
        aggfunc='count'
    ).fillna(0)


    # Reindex the pivot table to have the INWORKS_Category in the desired order
    ordered_categories = ['<90 days','90-180 days', '180-360 days', '>360 days']
    inworks_pivot = inworks_pivot.reindex(ordered_categories)


    closed_pivot = df[df['A_PRIMARY_STATUS'] == 'CLOSED'].pivot_table(
        index='CLOSED_Category',
        columns='Sortable Month-Year',
        values='A_PK_ID',
        aggfunc='count'
    ).fillna(0)
    # Reindex the pivot table to have the INWORKS_Category in the desired order
    ordered_categories1 = ['<90 days','90-180 days', '180-360 days', '>360 days']
    closed_pivot = closed_pivot.reindex(ordered_categories1)

    # Reorder the columns based on the 'Sortable Month-Year' key in ascending order
    inworks_pivot = inworks_pivot.reindex(columns=sorted(inworks_pivot.columns))
    closed_pivot = closed_pivot.reindex(columns=sorted(closed_pivot.columns))

    # Rename the columns to 'Month-Year' for display purposes
    inworks_pivot.columns = [pd.to_datetime(date, format='%Y%m').strftime('%b %Y') for date in inworks_pivot.columns]
    closed_pivot.columns = [pd.to_datetime(date, format='%Y%m').strftime('%b %Y') for date in closed_pivot.columns]



    # Add a Total column
    inworks_pivot['Total'] = inworks_pivot.sum(axis=1)
    closed_pivot['Total'] = closed_pivot.sum(axis=1)

    # Calculate the total number of INWORKS status entries for the percentage calculation
    total_inworks = inworks_pivot['Total'].sum()

    # Add a Percentage column for INWORKS pivot
    inworks_pivot['Percentage'] = (inworks_pivot['Total'] / total_inworks).mul(100).round(2)

    # Calculate the total number of CLOSED status entries for the percentage calculation
    total_closed = closed_pivot['Total'].sum()

    # Add a Percentage column only for categories that are not None
    closed_pivot['Percentage'] = (closed_pivot['Total'] / total_closed).mul(100).round(2)

    closed_pivot = closed_pivot[closed_pivot.index.notnull()]



    return inworks_pivot, closed_pivot


import plotly.graph_objs as go


# This function assumes that you have a DataFrame with columns 'Month-Year', 'Opened', and 'Closed'
def create_combined_bar_chart(df):
    # Convert 'Month-Year' to datetime to ensure proper sorting
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b %Y')
    # Sort the DataFrame by 'Month-Year'
    df.sort_values('Month-Year', inplace=True)

    # Create a pivot table with the counts of 'INWORKS' and 'CLOSED'
    status_counts = df.pivot_table(index='Month-Year', columns='A_PRIMARY_STATUS', values='A_PK_ID', aggfunc='count',
                                   fill_value=0)

    # Prepare the data for the bar chart
    inworks_counts = status_counts.get('INWORKS', [0] * len(status_counts))
    closed_counts = status_counts.get('CLOSED', [0] * len(status_counts))
    months = [date.strftime('%b %Y') for date in status_counts.index]

    # Create the bar chart using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Opened (INWORKS)', x=months, y=inworks_counts, marker_color='blue'))
    fig.add_trace(go.Bar(name='Closed', x=months, y=closed_counts, marker_color='green'))

    # Update the layout of the figure
    fig.update_layout(barmode='group', title='Opened vs Closed by Month',
                      xaxis_title='Month', yaxis_title='Count',
                      xaxis=dict(tickmode='array', tickvals=months, ticktext=months),
                      legend_title='Status',legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.02,  # Position legend above the figure
            xanchor="right",
            x=1
        ),
                      width=2000,  # Set the figure width
                      height=450   # Set the figure height
                      )

    return fig

def total_bar_chart(df):
    # Your actual DataFrame 'df' will be used here
    # ... (your DataFrame loading and processing)

    # Convert 'Month-Year' to datetime to ensure proper sorting
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%b %Y')

    # Sort the DataFrame by 'Month-Year'
    df.sort_values('Month-Year', inplace=True)

    # Create a pivot table with the counts of 'INWORKS' and 'CLOSED'
    status_counts = df.pivot_table(index='Month-Year', columns='A_PRIMARY_STATUS', values='A_PK_ID', aggfunc='count',
                                   fill_value=0)

    # Since we want to display only the total columns, we will calculate the total for each status.
    inworks_total = status_counts['INWORKS'].sum()
    closed_total = status_counts['CLOSED'].sum()

    # Now we'll create a bar chart to compare the totals using Plotly.
    fig = go.Figure()

    # Adding the total counts to the bar chart.
    fig.add_trace(go.Bar(name='Total INWORKS', x=['INWORKS'], y=[inworks_total], marker_color='blue'))
    fig.add_trace(go.Bar(name='Total CLOSED', x=['CLOSED'], y=[closed_total], marker_color='green'))

    # Update the layout of the figure
    fig.update_layout(
        title='Total INWORKS vs Total CLOSED',
        xaxis_title='Status',
        yaxis_title='Total Count',
        legend_title='Status',
        barmode='group',
        width=800,  # Set the figure width
        height=600  # Set the figure height
    )

    # Show the figure
    return fig



# Streamlit UI
st.title('Pivot Table Generator')

# Upload files
uploaded_files = st.file_uploader("Upload your Excel files", type=['xlsx'], accept_multiple_files=True)

# Sidebar filters
# Streamlit UI for date range selection
st.sidebar.title('Select Filters For All Document')
st.sidebar.title('Select Filters')
status_options = st.sidebar.multiselect('Select Status', options=['CURRENT', 'EXPIRED', 'INWORKS'], default=['CURRENT','EXPIRED'])
type_options = st.sidebar.multiselect('Select Type', options=['SOP', 'WI', 'OPI', 'QDL', 'PDL'], default=['SOP', 'WI', 'OPI', 'QDL', 'PDL'])

if uploaded_files and len(uploaded_files) == 2:
    combined_data = pd.DataFrame()

    for uploaded_file in uploaded_files:
        df = load_data(uploaded_file)
        combined_data = pd.concat([combined_data, df], ignore_index=True)

    # Assuming you have a 'Country' column in your DataFrame
    country_options = combined_data['Country'].unique()
    selected_countries = st.sidebar.multiselect('Select Country', country_options, default=country_options)

    # Filter the DataFrame based on selected type codes and countries
    combined_data = combined_data[combined_data['Country'].isin(selected_countries)]

    # Create a pivot table from the combined data
    combined_pivot_table = create_pivot_table(combined_data, status_options, type_options)

    # Get year ranges for total columns
    year_ranges = [get_year_range(load_data(file)) for file in uploaded_files]

    # Calculate and append totals for each year range
    for year_range in year_ranges:
        year_data = combined_data[combined_data['Effective Date'].dt.strftime('%Y').isin(year_range.split('-'))]
        # This should print out an array of unique years as integers
        print(year_data['Effective Date'].dt.year.unique())  # Check if 2023 is included
        year_counts = year_data['Effective Date'].dt.year.value_counts()
        year_pivot = create_pivot_table(year_data, status_options, type_options)
        combined_pivot_table[f'Total {year_range}'] = year_pivot.sum(axis=1)

    # Calculate and append the combined total
    #combined_pivot_table['Combined Total'] = combined_pivot_table.sum(axis=1)




    # Display combined pivot table
    st.dataframe(combined_data)
    st.write('Combined Pivot Table:')
    st.dataframe(combined_pivot_table)
    # Extract the most recent year and its data for the bar chart
    most_recent_year = combined_data['Effective Date'].dt.year.max()
    recent_year_data = combined_pivot_table[
        #for previous result[col for col in combined_pivot_table.columns if str(most_recent_year) in col and 'Total 2022' not in col]]
        [col for col in combined_pivot_table.columns if 'Total' in col and col.startswith('Total')]]
    monthly_totals = recent_year_data.sum().astype(int)

    # Create an interactive bar chart with Plotly
    st.write(f'Interactive Bar Chart for {most_recent_year}')
    fig = px.bar(
        x=monthly_totals.index,
        y=monthly_totals.values,
        labels={'x': 'Month', 'y': 'Total Count'},
        title=f'Total Count by Month for {most_recent_year}'
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('Please upload exactly two Excel files.')


# Streamlit UI
st.title('Pivot Table for Status Durations')

uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

if uploaded_file:
        # Load the data
        df = pd.read_excel(uploaded_file)
        # Extract only the numbers from the 'Owner' column
        df['OWNER_NUMBER'] = df['A_OWNER_CODE'].str.extract('(\d+)')
        country_codes = {
            '11': 'Sonova AG',
            '13': 'Sonova Communications AG',
            "3250": 'RCSE',
            '51': 'Sonova USA',
            '52': 'Sonova Canada',
            '53': 'Unitron',
            '6255': 'Mexicali',
            '74': 'SOCC',
            '84': 'SOCV'
        }

        # Map the country codes to the 'Owner_Number' column to create a new 'Country' column
        df['COUNTRY'] = df['OWNER_NUMBER'].map(country_codes)

        st.sidebar.title('Select Filters For All Change Plans')
        st.sidebar.title('Filter by Type Code')
        # Get unique values from the 'A_TYPE_CODE' column for the selection options
        type_code_options = df['A_TYPE_CODE'].unique()
        # Multiselect widget to choose type codes
        selected_type_codes = st.sidebar.multiselect('Select A_TYPE_CODE', type_code_options, default=type_code_options)

        # Assuming you have a 'Country' column in your DataFrame
        country_options = df['COUNTRY'].unique()
        selected_countries = st.sidebar.multiselect('Select Country', country_options, default=country_options)

        # Filter the DataFrame based on selected type codes and countries
        df= df[df['A_TYPE_CODE'].isin(selected_type_codes) & df['COUNTRY'].isin(selected_countries)]

        st.dataframe(df)


        # Process the DataFrame to calculate the pivot tables
        inworks_pivot, closed_pivot = create_status_pivot(df)

        # Display the pivot tables
        st.subheader('Time in INWORKS Status:')
        st.dataframe(inworks_pivot)

        st.subheader('Time to CLOSE Status:')
        st.dataframe(closed_pivot)

        st.subheader('A Type Of Code Pivot Table')
        #Make sure the date column is in datetime format
        df['Date'] = pd.to_datetime(df['Created Date'])



        #Create a pivot table
        pivot_table = df.pivot_table(
             index='A_TYPE_CODE',
             columns=df['Date'].dt.to_period('M'),
             # Replace with a column to aggregate, if needed
             aggfunc='size'  # or sum, mean, etc., depending on what you're measuring
         ).fillna(0)

        # Transform the column names to the desired format 'Nov 2019'
        pivot_table.columns = pivot_table.columns.strftime('%b %Y')

        # Add a Total column
        pivot_table['Total'] = pivot_table.sum(axis=1)


        # Calculate the total number of INWORKS status entries for the percentage calculation
        total_pivot = pivot_table['Total'].sum()

        # Add a Percentage column for INWORKS pivot
        pivot_table['Percentage'] = (pivot_table['Total'] / total_pivot).mul(100).round(2)
        st.dataframe(pivot_table)


        # Streamlit UI for date range selection
        st.sidebar.title('Select Date Range for Charts')
        min_date = df['Created Date'].min().to_pydatetime()  # Get the minimum date from your DataFrame
        max_date = df['Created Date'].max().to_pydatetime()  # Get the maximum date from your DataFrame

        # Set a different start date as default while allowing earlier dates
        default_start_date = min_date + pd.DateOffset(months=20)  # Example: One month after the min_date
        default_end_date = max_date

        # User selects the date range
        start_date = st.sidebar.date_input('Start date', default_start_date, min_value=min_date)
        end_date = st.sidebar.date_input('End date', default_end_date, min_value=min_date)

        # Filter the DataFrame based on the selected date range
        filtered_df = df[
            (df['Created Date'] >= pd.to_datetime(start_date)) & (df['Created Date'] <= pd.to_datetime(end_date))]

        # Filter for the months within the selected date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS').strftime('%b %Y').tolist()
        inworks_filtered = inworks_pivot.reindex(columns=date_range, fill_value=0).sum(axis=0)
        closed_filtered = closed_pivot.reindex(columns=date_range, fill_value=0).sum(axis=0)





        # Create bar charts using Plotly
        # Opened Chart
        fig_opened = px.bar(x=date_range, y=inworks_filtered.values, labels={'x': 'Month', 'y': 'Count'},
                            title='Opened')
        fig_opened.update_layout(xaxis_title='Month', yaxis_title='Opened Count')

        # Closed Chart
        fig_closed = px.bar(x=date_range, y=closed_filtered.values, labels={'x': 'Month', 'y': 'Count'}, title='Closed')
        fig_closed.update_layout(xaxis_title='Month', yaxis_title='Closed Count')

        # Reset index to convert the index into a column for Plotly
        pivot_table_example = pivot_table.reset_index().rename(columns={'index': 'A_TYPE_CODE'})

        # Melt the DataFrame to have A_TYPE_CODE, months, and values in separate columns
        pivot_melted = pivot_table_example.melt(id_vars=['A_TYPE_CODE'], var_name='Month', value_name='Value')

        # Create a bar plot with Plotly Express
        fig_code = px.bar(pivot_melted, x='Month', y='Value', color='A_TYPE_CODE', title='A Type Of Code Categories per Month')

        # Update the layout to adjust the bar mode
        fig_code.update_layout(barmode='group',
                          bargap=0.15,
                          yaxis=dict(range=[0, 200])  # Adjust this range as needed
                          )


        # Create the bar chart with Plotly Express
        # Now, create the bar chart with Plotly Express using the reset pivot table
        fig_total = px.bar(pivot_table_example, x='A_TYPE_CODE', y='Total', title='Total Counts by Type Code')

        # Streamlit - Display the charts
        st.plotly_chart(fig_opened)
        st.plotly_chart(fig_closed)

        combined_fig = create_combined_bar_chart(filtered_df)
        st.plotly_chart(combined_fig, use_container_width=True)

        total_fig= total_bar_chart(df)
        st.plotly_chart(total_fig, use_container_width=True)

        # Streamlit - Display the chart of A type of code column
        st.plotly_chart(fig_code)

        # Show the figure
        st.plotly_chart(fig_total)



else:
        st.info('Awaiting the upload of an Excel file.')

