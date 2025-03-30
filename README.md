# La Liga Dashboard

A luxurious interactive dashboard for visualizing and analyzing Spanish La Liga football data from 1995 to 2024.

## Features

- **Season Overview**: Interactive season standings, goal trends, and team goal production analysis
- **Team Analysis**: Detailed team performance metrics, radar charts for comparing performance across seasons, head-to-head analysis
- **Match Statistics**: First half vs second half goal analysis, match outcome distributions, team scoring patterns
- **Predictions**: Match outcome prediction model with win probability gauges and expected goals visualization

## Screenshots

[Screenshots would be added here]

## Requirements

- Python 3.9+
- Streamlit
- Pandas
- Plotly
- And other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Make sure the `LaLiga_Matches.csv` file is in the same directory as the application
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. The dashboard will launch in your default web browser

## Data

The dashboard uses La Liga match data from the 1995-1996 season to the 2023-2024 season. The dataset includes:

- Match date and teams
- Full-time and half-time scores
- Match results (home win, away win, or draw)

## Customization

You can easily customize the dashboard by:
- Adjusting the color theme in the CSS section
- Adding new visualizations to any tab
- Modifying the prediction model parameters

## License

MIT license

## Credits

Created with Streamlit, Plotly, and Pandas
