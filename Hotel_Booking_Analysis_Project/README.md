# Project — Hotel Booking & Occupancy Analysis

## Objective
Analyze hotel reservations, room rates, booking channels, cancellations, stay duration and revenue. The project addresses all 15 tasks in the assignment brief.

## Dataset
Place `13_hotel_bookings.csv` in this project folder. The assignment screenshot identifies this as the expected dataset filename. The CSV itself was not included with the screenshots, so the program reads the dataset you download from the repository.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python hotel_booking_analysis.py --input 13_hotel_bookings.csv
```

Optional output folder:
```bash
python hotel_booking_analysis.py --input 13_hotel_bookings.csv --output results
```

## Assignment task coverage
1. Loads and inspects records (shape, head, dtypes, missing values).
2. Reports missing values and duplicate Booking_ID rows; removes duplicate IDs.
3. Fills missing lead time and room rate using their medians.
4. Converts check-in date to datetime and extracts arrival month and weekday.
5. Calculates bookings, confirmed room-nights and cancellation percentage.
6. Calculates row-wise revenue as nights × room rate for non-cancelled bookings.
7. Compares average rates, room-nights and revenue by room type.
8. Ranks channels by booking volume and revenue.
9. Analyzes cancellations by channel and lead-time category.
10. Finds the top three revenue months.
11. Compares average stay duration by traveler type if that column exists.
12. Uses NumPy for mean, median and standard deviation of room rates.
13. Detects high booking-revenue outliers using IQR and `np.percentile()`.
14. Creates lead-time bands using `pd.cut()`.
15. Writes five management recommendations based on the results.

## Important interpretation notes
- Realized revenue excludes cancelled reservations.
- The script uses median imputation for missing lead-time and room-rate values and records the chosen medians in `summary_metrics.csv`.
- A true occupancy percentage requires room inventory / available-room data. If the CSV has no inventory column, the monthly chart uses confirmed room-nights as an occupancy-volume proxy and does not label it as a true occupancy percentage.
- The script detects common column-name variations. If a required field uses a different name, add that name to the relevant list in `ALIASES` near the top of the Python file.
- If the dataset does not include a traveler-type column, the leisure/business comparison is explicitly marked unavailable rather than inventing a result.
