def add_size_to_sample(orders_received_df, sample, axis=1):
    def add_size_to_canceled(row):
        if row.reason != 'canceled':
            return None
        mask = (orders_received_df.order_id == row.order_id)
        if mask.any():
            return orders_received_df[mask].iloc[0]['size']
    return sample.apply(add_size_to_canceled, axis=axis)
