from collections import defaultdict

import pandas as pd
import numpy as np


def add_size_to_sample(
        orders_received_df: pd.DataFrame,
        sample: pd.DataFrame,
        axis: int = 1
) -> pd.DataFrame:
    def add_size_to_canceled(row: pd.Series) -> pd.Series:
        if row.reason != 'canceled':
            return None
        mask = (orders_received_df.order_id == row.order_id)
        if mask.any():
            return orders_received_df[mask].iloc[0]['size']
    return sample.apply(add_size_to_canceled, axis=axis)


def cancelled_rolling_sum(
        df: pd.DataFrame,
        window_size: int,
        window_column: str,
        input_column: str,
        output_column: str,
        sort_by: str
) -> pd.DataFrame:
    df = df.sort_values(by=sort_by)
    left_ptr, right_ptr = 0, 0
    counter = 0
    timestamps = df.loc[:, window_column].to_list()
    values = df.loc[:, input_column].to_list()
    reason = df.loc[:, 'reason'].to_list()
    counter_col = []

    while right_ptr < df.shape[0]:
        l_timestamp = timestamps[left_ptr]
        r_timestamp = timestamps[right_ptr]
        if (left_ptr == right_ptr) or (r_timestamp - l_timestamp < window_size):
            condition = (reason[right_ptr] == 'canceled' and not np.isnan(values[right_ptr]))
            counter += values[right_ptr] if condition else 0
            counter_col.append(counter)
            right_ptr += 1
        else:
            condition = (reason[left_ptr] == 'canceled' and not np.isnan(values[left_ptr]))
            counter -= values[left_ptr] if condition else 0
            left_ptr += 1
    df.loc[:, output_column] = counter_col
    return df


def trade_rolling_sum(
        df: pd.DataFrame,
        window_size: int,
        window_column: str,
        input_column: str,
        output_column: str,
        sort_by: str
) -> pd.DataFrame:
    df = df.sort_values(by=sort_by)
    left_ptr, right_ptr = 0, 0
    counter = 0
    timestamps = df.loc[:, window_column].to_list()
    values = df.loc[:, input_column].to_list()
    counter_col = []

    while right_ptr < df.shape[0]:
        l_timestamp = timestamps[left_ptr]
        r_timestamp = timestamps[right_ptr]
        if (left_ptr == right_ptr) or (r_timestamp - l_timestamp < window_size):
            condition = not np.isnan(values[right_ptr])
            counter += 1 if condition else 0
            counter_col.append(counter)
            right_ptr += 1
        else:
            condition = not np.isnan(values[left_ptr])
            counter -= 1 if condition else 0
            left_ptr += 1
    df.loc[:, output_column] = counter_col
    return df


def value_change_rolling(
        df: pd.DataFrame,
        window_size: int,
        window_column: str,
        input_column: str,
        output_column: str,
        sort_by: str
):
    df = df.sort_values(by=sort_by)
    left_ptr, right_ptr = 0, 0
    timestamps = df.loc[:, window_column].to_list()
    values = df.loc[:, input_column].to_list()
    value_change_col = []

    while right_ptr < df.shape[0]:
        l_timestamp = timestamps[left_ptr]
        r_timestamp = timestamps[right_ptr]
        if (left_ptr == right_ptr) or (r_timestamp - l_timestamp < window_size):
            if values[left_ptr] != 0:
                value_change = (values[left_ptr] + (values[right_ptr] - values[left_ptr])) / values[left_ptr]
                value_change_col.append(value_change)
            else:
                value_change_col.append(0)
            right_ptr += 1
        else:
            left_ptr += 1
    df.loc[:, output_column] = value_change_col
    return df


def model_order_book(orders_df, output_start: int | None = None):
    buy_order_book = defaultdict(int)
    sell_order_book = defaultdict(int)
    received_orders = set()
    opened_orders = set()
    _min = None
    _max = None
    output_start = orders_df.index[-1] + 1 if output_start is None else output_start

    bid_volume = 0
    bid_volume_list = []
    ask_volume = 0
    ask_volume_list = []

    for i, (idx, row) in enumerate(orders_df.iterrows()):
        bid_volume_list.append(bid_volume)
        ask_volume_list.append(ask_volume)
        
        if i % 1000 == 0:
            print(f'{i} / {len(orders_df)}')
        _type, order_type, reason, order_id, maker_order_id, taker_order_id, price, size, remaining_size, old_size, new_size, side, best_bid, best_ask, timestamp = row['type'], row['order_type'], row['reason'], row['order_id'], row['maker_order_id'], row['taker_order_id'], row['price'], row['size'], row['remaining_size'], row['old_size'], row['new_size'], row['side'], row['best_bid'], row['best_ask'], row['timestamp']

        if i > output_start:
            print(idx)
            print(order_id)
        if order_type == 'market':
            received_orders.add(order_id)
            continue
        if _type == 'received':
            received_orders.add(order_id)
        elif _type == 'open':
            assert ~np.isnan(price)
            assert ~np.isnan(remaining_size)
            opened_orders.add(order_id)
            if side == 'buy':
                if _min is not None and price >= _min:
                    raise ValueError(f"Price {price} is greater than or equal to _min {_min}. {idx}")
                buy_order_book[price] += remaining_size
                bid_volume += remaining_size
            elif side == 'sell':
                if _max is not None and price <= _max:
                    raise ValueError(f"Price {price} is less than or equal to _max {_max}. {idx}")
                sell_order_book[price] += remaining_size
                ask_volume += remaining_size
            else:
                raise ValueError(f"Invalid side: {side}")
        elif _type == 'match':
            if maker_order_id not in received_orders:
                if i > output_start:
                    print(f'maker_order_id {maker_order_id} not in received_orders. skip')
                continue
            if taker_order_id not in received_orders:
                if i > output_start:
                    print(f'taker_order_id {taker_order_id} not in received_orders. skip')
                continue
            assert ~np.isnan(price)
            assert ~np.isnan(size)
            if side == 'buy':
                buy_order_book[price] -= size
                bid_volume -= size
                if np.isclose(buy_order_book[price], 0):
                    del buy_order_book[price]
            elif side == 'sell':
                sell_order_book[price] -= size
                ask_volume -= size
                if np.isclose(sell_order_book[price], 0):
                    del sell_order_book[price]
            else:
                raise ValueError(f"Invalid side: {side}")
        elif _type == 'change':
            if order_id not in received_orders:
                if i > output_start:
                    print(f'order_id {order_id} not in received_orders. skip')
                continue
            if order_id not in opened_orders:
                if i > output_start:
                    print(f'order_id {order_id} not in opened_orders. skip')
                continue
            assert ~np.isnan(new_size)
            assert ~np.isnan(old_size)
            diff = new_size - old_size
            if side == 'buy':
                buy_order_book[price] += diff
                bid_volume += diff
                if np.isclose(buy_order_book[price], 0):
                    del buy_order_book[price]
            elif side == 'sell':
                sell_order_book[price] += diff
                ask_volume += diff
                if np.isclose(sell_order_book[price], 0):
                    del sell_order_book[price]
            else:
                raise ValueError(f"Invalid side: {side}")
        elif reason == 'canceled' or _type == 'done':
            if order_id not in received_orders:
                if i > output_start:
                    print('skip. Order not in received')
                continue
            received_orders.remove(order_id)
            if order_id not in opened_orders:
                if i > output_start:
                    print('skip. Order not in opened')
                continue
            opened_orders.remove(order_id)
            if np.isnan(price) or np.isnan(remaining_size):
                continue
            if side == 'buy':
                buy_order_book[price] -= remaining_size
                bid_volume -= remaining_size
                if np.isclose(buy_order_book[price], 0):
                    del buy_order_book[price]
            elif side == 'sell':
                sell_order_book[price] -= remaining_size
                ask_volume -= remaining_size
                if np.isclose(sell_order_book[price], 0):
                    del sell_order_book[price]
            else:
                raise ValueError(f"Invalid side: {side}")
        _min = min(sell_order_book.keys()) if sell_order_book else None
        _max = max(buy_order_book.keys()) if buy_order_book else None
        if i > output_start:
            print('buy_order_book', buy_order_book)
            print('sell_order_book', sell_order_book)
    return buy_order_book, sell_order_book, best_bid, best_ask, bid_volume_list, ask_volume_list
