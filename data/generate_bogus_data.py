import csv
import random
import uuid
from datetime import datetime, timedelta

# --- Configuration ---
NUM_WATCH_SESSIONS = 500
TICKS_PER_SESSION = 600  # 10 minutes of data at 1 tick/sec
WATCH_DURATION_MINUTES = 10

WATCH_TRACKING_FILE = 'data/watch_tracking.csv'
PRICE_HISTORY_FILE = 'data/price_history.csv'

def generate_bogus_data():
    """
    Generates bogus data for watch_tracking.csv and price_history.csv.
    """
    print(f"Generating {NUM_WATCH_SESSIONS} watch sessions...")

    with open(WATCH_TRACKING_FILE, 'w', newline='') as watch_file, \
         open(PRICE_HISTORY_FILE, 'w', newline='') as price_file:

        watch_writer = csv.writer(watch_file)
        price_writer = csv.writer(price_file)

        # Write headers
        watch_writer.writerow(['perc_change', 'high_perc', 'low_perc', 'age', 'liq', 'mcap', 'vol', 'watch_start_time', 'watch_end_time', 'watch_id'])
        price_writer.writerow(['watch_id', 'timestamp', 'price', 'perc_change_from_entry'])

        start_time = datetime.now()

        for i in range(NUM_WATCH_SESSIONS):
            watch_id = str(uuid.uuid4())
            
            # Stagger start times
            session_start_time = start_time + timedelta(minutes=i * (WATCH_DURATION_MINUTES + 5))
            session_end_time = session_start_time + timedelta(minutes=WATCH_DURATION_MINUTES)

            # Generate session characteristics
            age = random.randint(30, 300)
            liq = round(random.uniform(50000, 6000000) / 10) * 10
            mcap = round((liq * random.uniform(1.0, 10.0)) / 10) * 10
            vol = round((mcap * random.uniform(0.1, 2.0)) / 10) * 10

            # Simulate price history for this session
            initial_price = random.uniform(0.0001, 0.05)
            prices = [initial_price]
            
            for j in range(TICKS_PER_SESSION):
                timestamp = session_start_time + timedelta(seconds=j)
                
                # Introduce a chance for the price to not change
                if j > 0 and random.random() < 0.7:  # 70% chance of price staying the same
                    current_price = prices[-1]
                else:
                    # Simple random walk for price
                    price_change_multiplier = random.uniform(-0.05, 0.05)
                    current_price = prices[-1] * (1 + price_change_multiplier)
                    if current_price <= 0: # prevent price from going to or below zero
                        current_price = prices[-1]

                prices.append(current_price)
                
                perc_change_from_entry = ((current_price - initial_price) / initial_price) * 100
                
                price_writer.writerow([
                    watch_id,
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    f"{current_price:.4f}",
                    f"{perc_change_from_entry:.4f}"
                ])
            
            # prices list includes the initial price, so we pop it for calculations on ticks after entry
            prices.pop(0)

            # Calculate stats for watch_tracking.csv
            entry_price = prices[0]
            final_price = prices[-1]
            high_price = max(prices)
            low_price = min(prices)

            perc_change = ((final_price - entry_price) / entry_price) * 100
            high_perc = ((high_price - entry_price) / entry_price) * 100
            low_perc = ((low_price - entry_price) / entry_price) * 100

            watch_writer.writerow([
                f"{perc_change:.2f}",
                f"{high_perc:.2f}",
                f"{low_perc:.2f}",
                age,
                f"{liq:.2f}",
                f"{mcap:.2f}",
                f"{vol:.2f}",
                session_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                session_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                watch_id
            ])

            if (i + 1) % 50 == 0:
                print(f"  ... generated {i + 1}/{NUM_WATCH_SESSIONS} sessions")

    print(f"\nSuccessfully generated data in:")
    print(f"- {WATCH_TRACKING_FILE}")
    print(f"- {PRICE_HISTORY_FILE}")

if __name__ == '__main__':
    generate_bogus_data() 