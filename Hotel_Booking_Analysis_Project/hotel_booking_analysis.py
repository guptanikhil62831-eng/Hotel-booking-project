"""
PROJECT 13: HOTEL BOOKING & OCCUPANCY ANALYSIS
------------------------------------------------
Run:
    pip install -r requirements.txt
    python hotel_booking_analysis.py --input 13_hotel_bookings.csv

Outputs are saved in the "outputs" folder:
    cleaned_bookings.csv
    summary_metrics.csv
    room_type_summary.csv
    booking_channel_summary.csv
    channel_cancellation_summary.csv
    monthly_revenue.csv
    monthly_occupancy_trend.csv
    traveler_stay_summary.csv
    lead_time_summary.csv
    revenue_outliers.csv
    daily_rate_statistics.csv
    recommendations.txt
    charts/*.png

Note:
True occupancy percentage requires room inventory / rooms available data.
If the dataset does not contain inventory, the monthly chart reports occupied
room-nights (an occupancy-volume proxy), not a true occupancy percentage.
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["#2E8B57", "#3CB371", "#66CDAA"])

warnings.filterwarnings("ignore", category=FutureWarning)

# Candidate column names make the project work with common dataset variations.
ALIASES = {
    "booking_id": ["Booking_ID", "BookingID", "booking_id", "Reservation_ID"],
    "lead_time": ["Lead_Time_Days", "Lead_Time", "lead_time", "LeadTime"],
    "room_rate": ["Room_Rate", "Daily_Room_Rate", "ADR", "Average_Daily_Rate", "room_rate"],
    "checkin": ["Checkin_Date", "Check_In_Date", "Arrival_Date", "checkin_date", "Date"],
    "nights": ["Nights", "Total_Nights", "Stay_Duration", "Total_Stay_Nights", "Number_of_Nights"],
    "weekend_nights": ["Weekend_Nights", "Stays_in_weekend_nights"],
    "weekday_nights": ["Weekday_Nights", "Stays_in_week_nights"],
    "room_type": ["Room_Type", "RoomType", "room_type"],
    "channel": ["Booking_Channel", "Market_Segment", "Channel", "booking_channel"],
    "cancelled": ["Is_Canceled", "Is_Cancelled", "Cancelled", "Canceled", "Cancellation_Status", "Booking_Status"],
    "traveler": ["Traveler_Type", "Guest_Type", "Customer_Type", "客人类型", "customer_type"],
    "inventory": ["Rooms_Available", "Available_Rooms", "Total_Rooms", "Room_Capacity"],
}

def find_column(df, logical_name, required=False):
    for candidate in ALIASES[logical_name]:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(
            f"Required column for '{logical_name}' was not found. "
            f"Accepted names: {ALIASES[logical_name]}\n"
            f"Columns present: {list(df.columns)}"
        )
    return None

def cancellation_flag(series):
    """Convert common cancellation labels / 0-1 values into True/False."""
    true_values = {"1", "true", "yes", "y", "cancelled", "canceled", "cancel"}
    false_values = {"0", "false", "no", "n", "not cancelled", "not canceled", "confirmed", "active"}
    def convert(value):
        if pd.isna(value):
            return np.nan
        s = str(value).strip().lower()
        if s in true_values:
            return True
        if s in false_values:
            return False
        # Common hotel datasets use 0 = not cancelled, 1 = cancelled.
        try:
            return bool(int(float(s)))
        except (ValueError, TypeError):
            return np.nan
    return series.map(convert)

def save_bar(data, x, y, title, xlabel, ylabel, path, rotate=25):
    if data.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(data[x].astype(str), data[y])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Hotel Booking & Occupancy Analysis")
    parser.add_argument("--input", default="13_hotel_bookings.csv",
                        help="Path to the hotel bookings CSV dataset")
    parser.add_argument("--output", default="outputs",
                        help="Folder in which results will be saved")
    args = parser.parse_args()

    input_path = Path(args.input)
    out = Path(args.output)
    charts = out / "charts"
    out.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    # 1. Load and inspect booking records.
    df = pd.read_csv(input_path)
    print("\n========== DATASET INSPECTION ==========")
    print("Shape:", df.shape)
    print("\nFirst five records:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())

    cols = {key: find_column(df, key, required=(key in
             ["booking_id", "lead_time", "room_rate", "checkin", "room_type", "channel", "cancelled"]))
            for key in ALIASES}

    # 2. Identify missing values and duplicate Booking_ID records.
    duplicate_ids = int(df[cols["booking_id"]].duplicated(keep=False).sum())
    missing_report = df.isna().sum().rename("Missing_Count").to_frame()
    missing_report["Missing_Percentage"] = (missing_report["Missing_Count"] / len(df) * 100).round(2)
    missing_report.to_csv(out / "missing_values_report.csv")
    duplicate_rows = df[df[cols["booking_id"]].duplicated(keep=False)].copy()
    duplicate_rows.to_csv(out / "duplicate_booking_ids.csv", index=False)

    # Remove duplicate booking IDs, keeping the first record.
    df = df.drop_duplicates(subset=[cols["booking_id"]], keep="first").copy()

    # 3. Handle missing Lead_Time_Days and Room_Rate with median imputation.
    df[cols["lead_time"]] = pd.to_numeric(df[cols["lead_time"]], errors="coerce")
    df[cols["room_rate"]] = pd.to_numeric(df[cols["room_rate"]], errors="coerce")
    lead_median = df[cols["lead_time"]].median()
    rate_median = df[cols["room_rate"]].median()
    if pd.isna(lead_median) or pd.isna(rate_median):
        raise ValueError("Lead time or room rate has no valid numeric values; median imputation is not possible.")
    df[cols["lead_time"]] = df[cols["lead_time"]].fillna(lead_median)
    df[cols["room_rate"]] = df[cols["room_rate"]].fillna(rate_median)

    # Convert numeric values and dates.
    df[cols["checkin"]] = pd.to_datetime(df[cols["checkin"]], errors="coerce")
    df = df.dropna(subset=[cols["checkin"]]).copy()
    df["Arrival_Month"] = df[cols["checkin"]].dt.month_name()
    df["Arrival_Year"] = df[cols["checkin"]].dt.year
    df["Arrival_Weekday"] = df[cols["checkin"]].dt.day_name()

    # Determine nights. Prefer a total-nights column; otherwise add weekday/weekend nights.
    if cols["nights"]:
        df["Total_Nights"] = pd.to_numeric(df[cols["nights"]], errors="coerce")
    elif cols["weekend_nights"] and cols["weekday_nights"]:
        df["Total_Nights"] = (
            pd.to_numeric(df[cols["weekend_nights"]], errors="coerce").fillna(0)
            + pd.to_numeric(df[cols["weekday_nights"]], errors="coerce").fillna(0)
        )
    else:
        raise ValueError("Could not find a nights column or both weekday/weekend nights columns.")
    df["Total_Nights"] = df["Total_Nights"].fillna(df["Total_Nights"].median()).clip(lower=0)

    # Normalize cancellation values.
    df["Is_Cancelled"] = cancellation_flag(df[cols["cancelled"]])
    unknown_cancel = int(df["Is_Cancelled"].isna().sum())
    # Unknown status is conservatively treated as not confirmed cancelled for counts,
    # but excluded from the cancellation-rate denominator.
    df["Is_Cancelled"] = df["Is_Cancelled"].fillna(False)

    # 4-5. Date features, booking count, room nights and cancellation percentage.
    total_bookings = len(df)
    cancelled_count = int(df["Is_Cancelled"].sum())
    cancellation_pct = (cancelled_count / total_bookings * 100) if total_bookings else 0
    total_room_nights = float(df.loc[~df["Is_Cancelled"], "Total_Nights"].sum())

    # 6. Realized revenue for non-cancelled bookings only.
    df["Booking_Revenue"] = np.where(
        ~df["Is_Cancelled"], df["Total_Nights"] * df[cols["room_rate"]], 0
    )

    # 7. Average room rates and room-night volume by room type.
    room_summary = df.groupby(cols["room_type"], dropna=False).agg(
        Booking_Count=(cols["booking_id"], "count"),
        Average_Room_Rate=(cols["room_rate"], "mean"),
        Confirmed_Room_Nights=("Total_Nights",
                               lambda s: s[~df.loc[s.index, "Is_Cancelled"]].sum()),
        Total_Revenue=("Booking_Revenue", "sum")
    ).reset_index().rename(columns={cols["room_type"]: "Room_Type"})
    room_summary[["Average_Room_Rate", "Total_Revenue"]] = room_summary[
        ["Average_Room_Rate", "Total_Revenue"]].round(2)
    room_summary.to_csv(out / "room_type_summary.csv", index=False)

    # 8. Rank booking channels by booking volume and revenue.
    channel_summary = df.groupby(cols["channel"], dropna=False).agg(
        Booking_Volume=(cols["booking_id"], "count"),
        Revenue=("Booking_Revenue", "sum"),
        Average_Room_Rate=(cols["room_rate"], "mean")
    ).reset_index().rename(columns={cols["channel"]: "Booking_Channel"})
    channel_summary = channel_summary.sort_values("Revenue", ascending=False)
    channel_summary[["Revenue", "Average_Room_Rate"]] = channel_summary[
        ["Revenue", "Average_Room_Rate"]].round(2)
    channel_summary.to_csv(out / "booking_channel_summary.csv", index=False)

    # 9. Cancellation rates by booking channel and lead-time bands.
    cancel_channel = df.groupby(cols["channel"]).agg(
        Bookings=(cols["booking_id"], "count"),
        Cancelled=("Is_Cancelled", "sum")
    ).reset_index().rename(columns={cols["channel"]: "Booking_Channel"})
    cancel_channel["Cancellation_Rate_Percent"] = (
        cancel_channel["Cancelled"] / cancel_channel["Bookings"] * 100
    ).round(2)
    cancel_channel.to_csv(out / "channel_cancellation_summary.csv", index=False)

    lead_bins = [-np.inf, 7, 30, 90, np.inf]
    lead_labels = ["Last-Minute (0-7)", "Short-Term (8-30)",
                   "Medium-Term (31-90)", "Advance (90+)"]
    df["Lead_Time_Category"] = pd.cut(
        df[cols["lead_time"]], bins=lead_bins, labels=lead_labels, right=True
    )
    lead_cancel = df.groupby("Lead_Time_Category", observed=False).agg(
        Bookings=(cols["booking_id"], "count"),
        Cancelled=("Is_Cancelled", "sum")
    ).reset_index()
    lead_cancel["Cancellation_Rate_Percent"] = (
        lead_cancel["Cancelled"] / lead_cancel["Bookings"].replace(0, np.nan) * 100
    ).round(2)
    lead_cancel.to_csv(out / "lead_time_cancellation_summary.csv", index=False)

    # 10. Top 3 peak revenue months.
    monthly_revenue = df.groupby(["Arrival_Year", "Arrival_Month"], sort=False)[
        "Booking_Revenue"].sum().reset_index()
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    monthly_revenue["Month_Number"] = pd.Categorical(
        monthly_revenue["Arrival_Month"], categories=month_order, ordered=True
    ).codes + 1
    monthly_revenue = monthly_revenue.sort_values(["Arrival_Year", "Month_Number"])
    monthly_revenue.to_csv(out / "monthly_revenue.csv", index=False)
    top_months = monthly_revenue.nlargest(3, "Booking_Revenue")
    top_months.to_csv(out / "top_3_peak_revenue_months.csv", index=False)

    # Monthly occupancy trend: use actual inventory if available, otherwise room-night proxy.
    if cols["inventory"]:
        df[cols["inventory"]] = pd.to_numeric(df[cols["inventory"]], errors="coerce")
        monthly_occ = df.groupby(["Arrival_Year", "Arrival_Month"], sort=False).agg(
            Occupied_Room_Nights=("Total_Nights",
                lambda s: s[~df.loc[s.index, "Is_Cancelled"]].sum()),
            Available_Rooms=(cols["inventory"], "max")
        ).reset_index()
        monthly_occ["Occupancy_Percent"] = (
            monthly_occ["Occupied_Room_Nights"] /
            (monthly_occ["Available_Rooms"] * monthly_occ["Total_Nights"].count()) * 100
        ).round(2)
        occupancy_label = "Occupancy_Percent"
    else:
        monthly_occ = df.groupby(["Arrival_Year", "Arrival_Month"], sort=False).apply(
            lambda g: pd.Series({
                "Confirmed_Room_Nights": g.loc[~g["Is_Cancelled"], "Total_Nights"].sum(),
                "Booking_Count": len(g)
            }), include_groups=False
        ).reset_index()
        occupancy_label = "Confirmed_Room_Nights (proxy)"
    monthly_occ.to_csv(out / "monthly_occupancy_trend.csv", index=False)

    # 11. Leisure vs business average stay duration, when traveler type is available.
    if cols["traveler"]:
        traveler_summary = df.groupby(cols["traveler"], dropna=False).agg(
            Average_Stay_Nights=("Total_Nights", "mean"),
            Booking_Count=(cols["booking_id"], "count")
        ).reset_index().rename(columns={cols["traveler"]: "Traveler_Type"})
        traveler_summary["Average_Stay_Nights"] = traveler_summary["Average_Stay_Nights"].round(2)
    else:
        traveler_summary = pd.DataFrame({
            "Note": ["Traveler-type column not present; leisure vs business comparison unavailable."]
        })
    traveler_summary.to_csv(out / "traveler_stay_summary.csv", index=False)

    # 12. NumPy mean, median and standard deviation of daily room rates.
    rates = df[cols["room_rate"]].dropna().to_numpy(dtype=float)
    rate_stats = pd.DataFrame([{
        "Mean_Daily_Room_Rate": np.mean(rates),
        "Median_Daily_Room_Rate": np.median(rates),
        "Std_Dev_Daily_Room_Rate": np.std(rates)
    }]).round(2)
    rate_stats.to_csv(out / "daily_rate_statistics.csv", index=False)

    # 13. Revenue outliers using IQR method and np.percentile().
    revenue_values = df.loc[~df["Is_Cancelled"], "Booking_Revenue"].to_numpy(dtype=float)
    q1, q3 = np.percentile(revenue_values, [25, 75]) if len(revenue_values) else (0, 0)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    outliers = df[(~df["Is_Cancelled"]) & (df["Booking_Revenue"] > upper_fence)].copy()
    outliers.to_csv(out / "revenue_outliers.csv", index=False)

    # 14. Lead-time categories have been created above with pd.cut().
    df.to_csv(out / "cleaned_bookings.csv", index=False)

    # Required charts.
    save_bar(room_summary, "Room_Type", "Total_Revenue",
             "Revenue by Room Type", "Room Type", "Revenue", charts / "revenue_by_room_type.png")
    save_bar(channel_summary, "Booking_Channel", "Booking_Volume",
             "Booking Volume by Channel", "Booking Channel", "Bookings",
             charts / "booking_volume_by_channel.png")
    save_bar(channel_summary, "Booking_Channel", "Revenue",
             "Revenue by Booking Channel", "Booking Channel", "Revenue",
             charts / "revenue_by_booking_channel.png")
    save_bar(cancel_channel, "Booking_Channel", "Cancellation_Rate_Percent",
             "Cancellation Rate by Booking Channel", "Booking Channel", "Cancellation (%)",
             charts / "cancellation_by_channel.png")
    save_bar(top_months.assign(Month=top_months["Arrival_Month"] + " " +
                               top_months["Arrival_Year"].astype(str)),
             "Month", "Booking_Revenue", "Top 3 Peak Revenue Months",
             "Month", "Revenue", charts / "top_peak_revenue_months.png")
    if occupancy_label in monthly_occ.columns:
        save_bar(monthly_occ.assign(Month=monthly_occ["Arrival_Month"] + " " +
                                    monthly_occ["Arrival_Year"].astype(str)),
                 "Month", occupancy_label, "Monthly Occupancy / Room-Night Trend",
                 "Month", occupancy_label, charts / "monthly_occupancy_trend.png")

    # Summary metrics.
    metrics = pd.DataFrame([
        ["Total unique bookings after duplicate removal", total_bookings],
        ["Duplicate Booking_ID rows detected", duplicate_ids],
        ["Missing cancellation statuses treated as not-cancelled", unknown_cancel],
        ["Total cancelled bookings", cancelled_count],
        ["Overall cancellation percentage", round(cancellation_pct, 2)],
        ["Confirmed room nights", round(total_room_nights, 2)],
        ["Realized booking revenue", round(df["Booking_Revenue"].sum(), 2)],
        ["Missing Lead_Time_Days filled with median", round(float(lead_median), 2)],
        ["Missing Room_Rate filled with median", round(float(rate_median), 2)],
        ["Revenue outliers detected by IQR", len(outliers)],
    ], columns=["Metric", "Value"])
    metrics.to_csv(out / "summary_metrics.csv", index=False)

    # 15. Five recommendations based on the analysis.
    recommendations = []
    if not channel_summary.empty:
        best = channel_summary.iloc[0]
        recommendations.append(
            f"1. Prioritize marketing partnerships and service quality for '{best['Booking_Channel']}', "
            f"the highest-revenue channel in this dataset (revenue {best['Revenue']:.2f})."
        )
    if not cancel_channel.empty:
        worst = cancel_channel.sort_values("Cancellation_Rate_Percent", ascending=False).iloc[0]
        recommendations.append(
            f"2. Review cancellation reasons and confirmation policies for '{worst['Booking_Channel']}', "
            f"which has the highest channel cancellation rate ({worst['Cancellation_Rate_Percent']:.2f}%)."
        )
    if not top_months.empty:
        peak = top_months.iloc[0]
        recommendations.append(
            f"3. Plan staffing, inventory and demand-based pricing ahead of {peak['Arrival_Month']} "
            f"{int(peak['Arrival_Year'])}, the highest-revenue month in the data."
        )
    if not room_summary.empty:
        room = room_summary.sort_values("Total_Revenue", ascending=False).iloc[0]
        recommendations.append(
            f"4. Monitor demand and guest feedback for '{room['Room_Type']}' rooms, "
            f"which contribute the most room-type revenue ({room['Total_Revenue']:.2f})."
        )
    recommendations.append(
        "5. Use lead-time cancellation patterns to test flexible deposits or reminder messages; "
        "measure cancellation changes before making the policy permanent."
    )
    (out / "recommendations.txt").write_text("\n".join(recommendations), encoding="utf-8")

    print("\n========== PROJECT COMPLETED ==========")
    print(metrics.to_string(index=False))
    print("\nTop 3 peak revenue months:\n", top_months.to_string(index=False))
    print("\nRecommendations:\n" + "\n".join(recommendations))
    print(f"\nAll files saved in: {out.resolve()}")
    if not cols["traveler"]:
        print("\nNOTE: Traveler type column was not found; leisure/business comparison is unavailable.")
    if not cols["inventory"]:
        print("NOTE: No room inventory column found; occupancy chart uses confirmed room-nights as a proxy.")

if __name__ == "__main__":
    main()
