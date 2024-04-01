# Inside data_processing.py

import pandas as pd

class DataProcessor:
    def __init__(self, data_path):
        """
        Initialize DataProcessor.

        Args:
        data_path (str): Path to the dataset CSV file.
        """
        self.data_path = data_path
        self.df = None

    def load_dataset(self):
        """
        Load the dataset.

        Returns:
        pandas.DataFrame: Loaded dataset.
        """
        self.df = pd.read_csv(self.data_path)
        return self.df

    def investigate_data(self):
        """
        Investigate the data.

        Prints descriptive statistics and top channels for uploads, subscribers, and views.
        """
        uploads = self.df['uploads'].dropna()
        count = uploads.sum()
        min_uploads = uploads.min()
        max_uploads = uploads.max()
        upload_range = max_uploads - min_uploads
        mean_upload = uploads.mean()
        median_upload = uploads.median()
        mode_uploads = uploads.mode().values[0]
        variance_uploads = uploads.var()
        std_dev_uploads = uploads.std()

        uploads_by_channel = self.df.groupby('Youtuber')['uploads'].sum().dropna()
        top_uploaders = uploads_by_channel.sort_values(ascending=False)

        subs_by_channel = self.df.groupby('Youtuber')['subscribers'].sum().dropna()
        top_subs = subs_by_channel.sort_values(ascending=False)

        views_by_channel = self.df.groupby('Youtuber')['video views'].sum().dropna()
        top_views = views_by_channel.sort_values(ascending=False)

        # Print descriptive statistics
        print(f"1. Count of uploads: {count}")
        print(f"2. Minimum uploads: {min_uploads:.2f}")
        print(f"3. Maximum uploads: {max_uploads:.2f}")
        print(f"4. Upload range: {upload_range:.2f}")
        print(f"5. Mean Upload: {mean_upload:.2f}")
        print(f"6. Median Upload: {median_upload:.2f}")
        print(f"7. Mode of Uploads: {mode_uploads:.2f}")
        print(f"8. Variance of Uploads: {variance_uploads:.2f}")
        print(f"9. Standard Deviation of Uploads: {std_dev_uploads:.2f}")

        print("\nChannels with the Most Uploads: ")
        print(top_uploaders)

        print("\nChannels with the Most Subscribers: ")
        print(top_subs)

        print("\nChannels with the Most Views: ")
        print(top_views)

        # Visualizing Raw Data
        print(self.df.head().transpose())
