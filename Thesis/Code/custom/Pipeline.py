import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

class Pipeline:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
    def __init__(self, classifier, regressor, mode, df, company):
        self.classifier = classifier
        self.regressor = regressor
        self.mode = mode
        self.df = df
        self.company = company
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.categorical_features = ['LAADPC', 'dayofweekcreation', 'weeknr']
        self.continuous_features = ['PALLETPLAATSEN', 'AANTALORDERS']

    def prepare_X_values(self, X):
        encoded_categorical_features = self.encoder.transform(X[self.categorical_features])
        scaled_continuous_features = self.scaler.transform(X[self.continuous_features])
        return np.hstack([scaled_continuous_features, encoded_categorical_features])
    
    def get_X_and_Y(self):
        region_columns = [col for col in self.df.columns if col.startswith('REGION_')]
        # Filter data for the specific company
        df_to_use = self.df

        # Calculate the totals for each region column and aquire targets
        totals = df_to_use[region_columns].sum()
        non_zero_totals = totals[totals != 0]
        targets = list(non_zero_totals.keys())
              
        encoded_categorical_features = self.encoder.fit_transform(df_to_use[self.categorical_features])
        scaled_continuous_features = self.scaler.fit_transform(df_to_use[self.continuous_features])

        # Combine all features
        X_formatted = np.hstack([scaled_continuous_features, encoded_categorical_features])
        Y = df_to_use[targets]  
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X_formatted, Y, test_size=0.2, random_state=42)
        self.Y_train_binary = self.Y_train.apply(lambda row: row.map(lambda x: 1 if x > 0 else 0), axis=1)
        self.Y_test_binary = self.Y_test.apply(lambda row: row.map(lambda x: 1 if x > 0 else 0), axis=1)

    def add_binary_to_X(self, X, Y_binary):
        return np.hstack([X, Y_binary])

    def get_original_palletplaatsen(self, X_scaled):
        original_values = self.scaler.inverse_transform(np.column_stack((X_scaled[:, 0], X_scaled[:, 1])))
        original_df = pd.DataFrame(original_values, columns=['PALLETPLAATSEN', 'AANTALORDERS'])
        return original_df["PALLETPLAATSEN"]
    

    def filter_by_binary(self, df_float, df_binary):
        df_float = pd.DataFrame(df_float, columns=self.Y_train.columns)
        # Ensure the dataframes have the same shape
        if df_float.shape != df_binary.shape:
            raise ValueError("The two dataframes must have the same shape.")
        
        # Apply the mask
        result_df = df_float.where(df_binary == 1, 0)
        return result_df
    
    def softmax_to_demand(self, df_predicted, given_demand):
        # Ensure given_demand is a Series for iteration
        df_predicted.reset_index(inplace=True, drop=True)
        given_demand.reset_index(inplace=True, drop=True)
        given_demand = given_demand.squeeze()  # This works if given_demand is a DataFrame with one column or a Series
        
        # Iterate over each row by index, assuming both df_predicted and given_demand use the same index
        for row_index in df_predicted.index:
            demand = given_demand.loc[row_index]
            row_values = df_predicted.loc[row_index]
            
            if row_values.sum() == 0:
                continue  # Skip rows where the sum is zero to avoid division by zero

            # Apply scaling factor only to non-zero elements
            non_zero_indices = row_values != 0
            non_zero_values = row_values[non_zero_indices]
            scaling_factor = demand / non_zero_values.sum()
            
            # Scale and apply softmax logic to non-zero values
            non_zero_adjusted = np.exp(non_zero_values * scaling_factor - np.max(non_zero_values * scaling_factor))
            df_predicted.loc[row_index, non_zero_indices] = np.round((non_zero_adjusted / non_zero_adjusted.sum()) * demand)

        return df_predicted

    def scale_to_demand(self, df_predicted, given_demand):
        df_predicted.reset_index(inplace=True, drop=True)
        given_demand.reset_index(inplace=True, drop=True)
        for row, demand in given_demand.items():  # assuming given_demand is a dict or series
            total_predicted = df_predicted.loc[row].sum()
            if total_predicted == 0:
                continue  # Skip if no demand is predicted to avoid division by zero
            scaling_factor = demand / total_predicted
            df_predicted.loc[row] = np.round(df_predicted.loc[row] * scaling_factor)
        return df_predicted
    

    
    # Prediction functions
    def predict_classifier(self, X):
        Y_pred_df = pd.DataFrame(self.classifier.predict(X), columns=self.Y_train_binary.columns)
        return Y_pred_df
    
    def predict_destination(self, X):
        inputvalues = self.prepare_X_values(X)
        return self.classifier.predict(inputvalues) 

    def predict_demands_series(self, X):
        inputvalues = self.prepare_X_values(X)
        pred_test_Y = self.predict_classifier(inputvalues)
        X = self.add_binary_to_X(inputvalues, pred_test_Y)
        Y_pred = self.regressor.predict(X)
        Y_pred = self.filter_by_binary(Y_pred, pred_test_Y)
        return Y_pred
    
    def predict_demands_parallel(self, X):
        Y_pred_bin = self.predict_classifier(X)
        Y_pred = self.regressor.predict(X)
        Y_pred_filtered = self.filter_by_binary(Y_pred, Y_pred_bin)
        return Y_pred_filtered
    
    def predict_demands(self, row):
        X = self.prepare_X_values(row)
        if self.mode == "parallel":
            return self.predict_demands_parallel(X)
        elif self.mode == "series":
            return self.predict_demands_series(X)
        
    def predict_demands_with_correction(self, row, softmax=False):
        df_predicted = self.predict_demands(row)
        OGdemand = row["PALLETPLAATSEN"]
        if softmax:
            return self.softmax_to_demand(df_predicted, OGdemand)
        else:
            return self.scale_to_demand(df_predicted, OGdemand)

    # Training functions
    def train_classifier(self):
        self.classifier.fit(self.X_train, self.Y_train_binary)

    def train_clean_regressor(self):
        X = self.add_binary_to_X(self.X_train, self.Y_train_binary)
        self.regressor.fit(X, self.Y_train)
        
    def train_dirty_regressor(self):
        pred_train_Y = self.predict_classifier(self.X_train)
        X = self.add_binary_to_X(self.X_train, pred_train_Y)
        self.regressor.fit(X, self.Y_train)

    def train_parallel_regressor(self):
        self.regressor.fit(self.X_train, self.Y_train)

    # Scoring functions
    def score_classifier(self):
        Y_pred = self.predict_classifier(self.X_test)
        zerodiv = 1
        accuracy = accuracy_score( self.Y_test_binary.values.flatten(), Y_pred.values.flatten())
        precision = precision_score( self.Y_test_binary,  Y_pred, average='macro', zero_division=zerodiv)
        recall = recall_score( self.Y_test_binary,  Y_pred, average='macro', zero_division=zerodiv)
        f1 = f1_score(self.Y_test_binary, Y_pred, average='macro', zero_division=zerodiv)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def score_clean_regressor(self):
        X = self.add_binary_to_X(self.X_test, self.Y_test_binary)
        Y_pred = self.regressor.predict(X)
        mse = mean_squared_error(self.Y_test, Y_pred)
        r2 = r2_score(self.Y_test, Y_pred)
        return {"mean_squared_error": mse, "r2_score": r2}
    
    def score_dirty_regressor(self):
        pred_test_Y = self.predict_classifier(self.X_test)
        X = self.add_binary_to_X(self.X_test, pred_test_Y)
        Y_pred = self.regressor.predict(X)
        mse = mean_squared_error(self.Y_test, Y_pred)
        r2 = r2_score(self.Y_test, Y_pred)
        return {"mean_squared_error": mse, "r2_score": r2}
    
    def score_parallel_regressor(self):
        Y_pred = self.regressor.predict(self.X_test)
        mse = mean_squared_error(self.Y_test, Y_pred)
        r2 = r2_score(self.Y_test, Y_pred)
        return {"mean_squared_error": mse, "r2_score": r2}

    def score_pipeline(self):
        pred_test_Y = self.predict_classifier(self.X_test)
        X = self.add_binary_to_X(self.X_test, pred_test_Y)
        Y_pred = self.regressor.predict(X)
        Y_pred = self.filter_by_binary(Y_pred, pred_test_Y)
        mse = mean_squared_error(self.Y_test, Y_pred)
        r2 = r2_score(self.Y_test, Y_pred)
        return {"mean_squared_error": mse, "r2_score": r2}
    
    def score_pipeline_with_correction(self):
        pred_test_Y = self.predict_classifier(self.X_test)
        X = self.add_binary_to_X(self.X_test, pred_test_Y)
        Y_pred = self.regressor.predict(X)
        Y_pred = self.filter_by_binary(Y_pred, pred_test_Y)
        original_palletplaatsen = self.get_original_palletplaatsen(self.X_test)
        Y_pred = self.scale_to_demand(Y_pred, original_palletplaatsen)
        mse = mean_squared_error(self.Y_test, Y_pred)
        r2 = r2_score(self.Y_test, Y_pred)
        return {"mean_squared_error": mse, "r2_score": r2}
    
    def score_pipeline_with_correction2(self):
        pred_test_Y = self.predict_classifier(self.X_test)
        X = self.add_binary_to_X(self.X_test, pred_test_Y)
        Y_pred = self.regressor.predict(X)
        Y_pred = self.filter_by_binary(Y_pred, pred_test_Y)
        original_palletplaatsen = self.get_original_palletplaatsen(self.X_test)
        Y_pred = self.softmax_to_demand(Y_pred, original_palletplaatsen)
        mse = mean_squared_error(self.Y_test, Y_pred)
        r2 = r2_score(self.Y_test, Y_pred)
        return {"mean_squared_error": mse, "r2_score": r2}
    
    def score_parallel_pipeline(self):
        Y_pred_bin = self.predict_classifier(self.X_test)
        Y_pred = self.regressor.predict(self.X_test)
        Y_pred_filtered = self.filter_by_binary(Y_pred, Y_pred_bin)
        mse = mean_squared_error(self.Y_test, Y_pred_filtered)
        r2 = r2_score(self.Y_test, Y_pred_filtered)
        return {"mean_squared_error": mse, "r2_score": r2}
    
    def score_parallel_pipeline_with_correction(self):
        Y_pred_bin = self.predict_classifier(self.X_test)
        Y_pred = self.regressor.predict(self.X_test)
        Y_pred_filtered = self.filter_by_binary(Y_pred, Y_pred_bin)
        original_palletplaatsen = self.get_original_palletplaatsen(self.X_test)
        Y_pred_scaled = self.scale_to_demand(Y_pred_filtered, original_palletplaatsen)
        mse = mean_squared_error(self.Y_test, Y_pred_scaled)
        r2 = r2_score(self.Y_test, Y_pred_scaled)
        return {"mean_squared_error": mse, "r2_score": r2}
    
    def score_parallel_pipeline_with_correction2(self):
        Y_pred_bin = self.predict_classifier(self.X_test)
        Y_pred = self.regressor.predict(self.X_test)
        Y_pred_filtered = self.filter_by_binary(Y_pred, Y_pred_bin)
        original_palletplaatsen = self.get_original_palletplaatsen(self.X_test)
        Y_pred_softmax = self.softmax_to_demand(Y_pred_filtered, original_palletplaatsen)
        mse = mean_squared_error(self.Y_test, Y_pred_softmax)
        r2 = r2_score(self.Y_test, Y_pred_softmax)
        return {"mean_squared_error": mse, "r2_score": r2}