import pandas as pd
import numpy as np

# Generate 2 years of daily dates
dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")

# Create sales with trend + weekly seasonality + noise
trend = np.linspace(120, 500, len(dates))  # upward trend
seasonality = 20 * np.sin(2 * np.pi * dates.dayofyear / 7)  # weekly cycle
noise = np.random.normal(0, 5, len(dates))  # small random noise

sales = trend + seasonality + noise

# Build DataFrame
df = pd.DataFrame({"date": dates, "sales": sales.astype(int)})

# Save to CSV
df.to_csv("data/daily_sales.csv", index=False)

print("✅ Dataset generated: data/daily_sales.csv with", len(df), "rows")