from matplotlib import pyplot as plt


def plot_cumulative_momentum(cumulative_momentum):
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_momentum)
    plt.xlabel('Timestamp')
    plt.ylabel('Cumulative Momentum')
    plt.title('Cumulative Momentum over Time')
    plt.show()


def plot_mid_price(mean_price):
    plt.figure(figsize=(10, 5))
    plt.plot(mean_price, label='Mid Price')
    plt.xlabel('Timestamp')
    plt.ylabel('Mid Price')
    plt.title('Mid Price over Time')
    plt.legend()
    plt.show()


def plot_mid_price_with_momentum(mean_price, cumulative_momentum):
    plt.figure(figsize=(10, 5))
    normalized_price = (mean_price - min(mean_price)) / (max(mean_price) - min(mean_price))
    normalized_momentum = (cumulative_momentum - min(cumulative_momentum)) / (max(cumulative_momentum) - min(cumulative_momentum))
    plt.plot(normalized_price, label='Mid Price (Normalized)')
    plt.plot(normalized_momentum, label='Cumulative Momentum (Normalized)') 
    plt.xlabel('Timestamp')
    plt.ylabel('Normalized Values')
    plt.title('Normalized Mid Price with Cumulative Momentum')
    plt.legend()
    plt.show()
