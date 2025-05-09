from concurrent.futures import ProcessPoolExecutor

import tqdm
import numpy as np
import pandas as pd


def calculate_buy_velocity(price: float, alpha: float, delta_t_ns: int, bid: float):
    return (price - (bid - alpha)) / delta_t_ns


def calculate_sell_velocity(price: float, alpha: float, delta_t_ns: int, ask: float):
    return (price - (ask + alpha)) / delta_t_ns


def calculate_buy_cancel_velocity(price: float, alpha: float, delta_t_ns: int, bid: float):
    return ((bid - alpha) - price) / delta_t_ns


def calculate_sell_cancel_velocity(price: float, alpha: float, delta_t_ns: int, ask: float):
    return ((ask + alpha) - price) / delta_t_ns


def calculate_momentum(orders_df, alpha, delta_t_ns, start_t_ns, end_t_ns, area='active'):
    accum_df = None

    cumulative_momentum = []
    mean_price = []

    for t in tqdm.tqdm(range(start_t_ns, end_t_ns, delta_t_ns)):
        time_mask = orders_df.timestamp.between(t, t + delta_t_ns, inclusive='left')
        if area == 'active':
            area_mask = orders_df.price.between(orders_df.best_bid - alpha, orders_df.best_ask + alpha)
            alpha_border = alpha
        elif area == 'passive':
            bid_area = orders_df.price.between(orders_df.best_bid - 2 * alpha, orders_df.best_bid - alpha)
            ask_area = orders_df.price.between(orders_df.best_ask + alpha, orders_df.best_ask + 2 * alpha)
            area_mask = bid_area | ask_area
            alpha_border = 2 * alpha
        else:
            raise ValueError(f"Invalid area: {area}")
        tmp_df = orders_df[time_mask & area_mask].copy()
        if tmp_df.shape[0] == 0:
            continue
    
        tmp_df['velocity'] = np.nan
        tmp_df['momentum'] = np.nan

        buy_mask = (tmp_df.order_type == 'limit') & (tmp_df.side == 'buy')
        sell_mask = (tmp_df.order_type == 'limit') & (tmp_df.side == 'sell')
        buy_cancel_mask = (tmp_df.reason == 'canceled') & (tmp_df.side == 'buy')
        sell_cancel_mask = (tmp_df.reason == 'canceled') & (tmp_df.side == 'sell')
        market_buy_mask = (tmp_df.order_type == 'market') & (tmp_df.side == 'buy')
        market_sell_mask = (tmp_df.order_type == 'market') & (tmp_df.side == 'sell')
        
        tmp_df.loc[buy_mask, 'velocity'] = calculate_buy_velocity(
            tmp_df.loc[buy_mask, 'price'],
            alpha_border,
            delta_t_ns / 1_000_000_000,
            tmp_df.loc[buy_mask, 'best_bid']
        )
        tmp_df.loc[buy_mask, 'momentum'] = tmp_df.loc[buy_mask, 'velocity'] * tmp_df.loc[buy_mask, 'size']

        tmp_df.loc[sell_mask, 'velocity'] = calculate_sell_velocity(
            tmp_df.loc[sell_mask, 'price'],
            alpha_border,
            delta_t_ns / 1_000_000_000,
            tmp_df.loc[sell_mask, 'best_ask']
        )
        tmp_df.loc[sell_mask, 'momentum'] = tmp_df.loc[sell_mask, 'velocity'] * tmp_df.loc[sell_mask, 'size']

        tmp_df.loc[buy_cancel_mask, 'velocity'] = calculate_buy_cancel_velocity(
            tmp_df.loc[buy_cancel_mask, 'price'],
            alpha_border,
            delta_t_ns / 1_000_000_000,
            tmp_df.loc[buy_cancel_mask, 'best_bid']
        )
        tmp_df.loc[buy_cancel_mask, 'momentum'] = tmp_df.loc[buy_cancel_mask, 'velocity'] * tmp_df.loc[buy_cancel_mask, 'size']

        tmp_df.loc[sell_cancel_mask, 'velocity'] = calculate_sell_cancel_velocity(
            tmp_df.loc[sell_cancel_mask, 'price'],
            alpha_border,
            delta_t_ns / 1_000_000_000,
            tmp_df.loc[sell_cancel_mask, 'best_ask']
        )
        tmp_df.loc[sell_cancel_mask, 'momentum'] = tmp_df.loc[sell_cancel_mask, 'velocity'] * tmp_df.loc[sell_cancel_mask, 'size']

        tmp_df.loc[market_buy_mask, 'velocity'] = calculate_buy_velocity(
            tmp_df.loc[market_buy_mask, 'best_ask'],
            alpha_border,
            delta_t_ns / 1_000_000_000,
            tmp_df.loc[market_buy_mask, 'best_bid']
        )
        tmp_df.loc[market_buy_mask, 'momentum'] = tmp_df.loc[market_buy_mask, 'velocity'] * tmp_df.loc[market_buy_mask, 'size']

        tmp_df.loc[market_sell_mask, 'velocity'] = calculate_sell_velocity(
            tmp_df.loc[market_sell_mask, 'best_bid'],
            alpha_border,
            delta_t_ns / 1_000_000_000,
            tmp_df.loc[market_sell_mask, 'best_ask']
        )
        tmp_df.loc[market_sell_mask, 'momentum'] = tmp_df.loc[market_sell_mask, 'velocity'] * tmp_df.loc[market_sell_mask, 'size']

        mean_price.append(tmp_df.mid_price.mean())
        net_momentum = tmp_df.momentum.sum()
        if len(cumulative_momentum) > 0:
            cumulative_momentum.append(net_momentum + cumulative_momentum[-1])
        else:
            cumulative_momentum.append(net_momentum)
        
        if accum_df is None:
            accum_df = tmp_df.copy()
        else:
            accum_df = pd.concat([accum_df, tmp_df])

    return cumulative_momentum, mean_price, accum_df


def calculate_momentum_multiproc(
        orders_df,
        alpha,
        delta_t_ns,
        start_t_ns,
        end_t_ns,
        area='active',
        concurrency=10,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    timestamp_interval = (end_t_ns - start_t_ns) // concurrency
    
    with ProcessPoolExecutor(concurrency) as executor:
        futures = []
        for start_timestamp in range(start_t_ns, end_t_ns, timestamp_interval):
            tmp_df = orders_df[orders_df.timestamp.between(
                start_timestamp,
                start_timestamp + timestamp_interval,
                inclusive='left'
            )]
            future = executor.submit(
                calculate_momentum,
                tmp_df,
                alpha,
                delta_t_ns,
                start_timestamp,
                start_timestamp + timestamp_interval,
                area=area
            )
            futures.append(future)

    cumulative_momentum, mean_price_full, accum_df = futures[0].result()
    shift = cumulative_momentum[-1]
    for future in futures[1:]:
        momentum, mean_price, df = future.result()
        mean_price_full.extend(mean_price)
        shifted_array = np.array(momentum) + shift
        cumulative_momentum.extend(shifted_array)
        shift = cumulative_momentum[-1]
        accum_df = pd.concat([accum_df, df])

    return np.array(cumulative_momentum), np.array(mean_price_full), accum_df
