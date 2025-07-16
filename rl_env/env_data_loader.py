import pandas as pd
import numpy as np
from datetime import timedelta

class SmartMeterDataLoader:
    def __init__(self, aggregate_load_segments: list, 
                 aggregate_load_df: pd.DataFrame, 
                 segment_length: int = 24):
        """
        Initializes the data loader with segments and DataFrame.
        
        :param aggregate_load_segments: Numpy array of shape (N, segment_length) containing load segments.
        :param aggregate_load_df: Pandas DataFrame containing the full load data.
        :param segment_length: Length of each segment in hours
        """
        self.aggregate_load_segments = aggregate_load_segments
        self.aggregate_load_df = aggregate_load_df
        self.segment_length = timedelta(hours=segment_length)

        self.divided_segments = self._divide_segments()


    def _divide_segments(self) -> np.ndarray:
        '''Divide the aggregate load segments into smaller segments of specified length.'''
        num_segments = len(self.aggregate_load_segments)
        divided_segments = []

        for i in range(num_segments):
            segment = self.aggregate_load_segments[i]
            start_time = segment[0]
            end_time = segment[1]
            # Create smaller segments within the specified length
            for j in pd.date_range(start=start_time, end=end_time, freq=self.segment_length):
                divided_segments.append([
                    j.to_pydatetime(), 
                    min(j + self.segment_length - timedelta(microseconds=1), end_time)
                ])

        return np.array(divided_segments)
    
    def get_divided_segments_length(self) -> int:
        """Returns the number of divided segments."""
        return self.divided_segments.shape[0]
    

    def get_aggregate_load_segment(self, index: int) -> pd.DataFrame:
        """Retrieves a specific segment by index.

        :param index: Index of the segment to retrieve.
        :return: Pandas DataFrame containing the segment data.
        """
        if index < 0 or index >= self.divided_segments.shape[0]:
            raise IndexError("Segment index out of bounds.")
        
        segment = self.divided_segments[index]

        # base on the segment, get the corresponding aggregate load data
        start_datetime, end_datetime = segment[0], segment[1]
        aggregate_load = self.aggregate_load_df[self.aggregate_load_df['datetime'].between(start_datetime, end_datetime)]

        # we need the timestamp to create datetime related features in the environment
        aggregate_load = aggregate_load[['timestamp', 'aggregate', 'datetime']]

        return aggregate_load