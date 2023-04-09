import sys     
import pandas as pd   
import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource

class MatchFunctions:
    def __init__(self):
        """
        A class that contains three methods to Identify_best_fits, search_ideal_match and prepare_graphs
        from a given set of data.
        """
        pass

    def Identify_best_fits(self, train_fun, ideal_fun):
        """
        Finds matches between training functions and ideal functions based on min(MSE)

        Args:
            train_fun (pd.DataFrame): A dataframe containing training functions.
            ideal_fun (pd.DataFrame): A dataframe containing ideal functions.

        Returns:
            pd.DataFrame: A dataframe with ideal functions and their deviations.
        
        Raises:
            TypeError: If the given arguments are not of DataFrame type.
        """
        if isinstance(train_fun, pd.DataFrame) and isinstance(ideal_fun, pd.DataFrame):
            ideal_lcol = len(ideal_fun.columns)
            train_lrow = train_fun.index[-1] + 1
            train_col = len(train_fun.columns)

            index_list = []  
            least_square = []  
            for j in range(1, train_col):
                least_square1 = []
                for k in range(1, ideal_lcol): 
                    MSE_sum = 0  
                    for i in range(train_lrow):
                        z1 = train_fun.iloc[i, j]  
                        z2 = ideal_fun.iloc[i, k]  
                        MSE_sum = MSE_sum + ((z1 - z2) ** 2)
                    least_square1.append(MSE_sum / train_lrow)
                min_least = min(least_square1)
                index = least_square1.index(min_least) 
                index_list.append(index + 1)
                least_square.append(min_least)

            per_frame = pd.DataFrame(list(zip(index_list, least_square)), columns=["Index", "least_square_value"])

            return per_frame
        else:
            raise TypeError("Given arguments are not of Dataframe type.")
    def search_ideal_match(self, test_fun):
        """
        Determine for each and every x-y-pair of values whether they can be assigned to the four chosen ideal functions

        Args:
        test_fun: pandas DataFrame with x and y values

        Returns:
        pandas DataFrame paired with values from the four ideal functions
        """
        if isinstance(test_fun, pd.DataFrame):
            test_lrow = test_fun.index[-1] + 1  
            test_lcol = len(test_fun.columns) 

            ideal_index = [] 
            deviation = [] 
            for j in range(test_lrow): 
                MSE_l = [] 
                for i in range(2, test_lcol): 
                    z1 = test_fun.iloc[j, 1]
                    z2 = test_fun.iloc[j, i]
                    MSE_sum = ((z2 - z1) ** 2)  
                    MSE_l.append(MSE_sum) 
                min_least = min(MSE_l)  
                if min_least < (np.sqrt(2)):
                    deviation.append(min_least) 
                    index = MSE_l.index(min_least) 
                    ideal_index.append(index) 
                else:
                    deviation.append(min_least)
                    ideal_index.append("Miss")  

            test["Deviation"] = deviation
            test["Ideal index"] = ideal_index

            return test

        else:
            raise TypeError("Given argument is not of Dataframe type.")

class BokehGraphs(MatchFunctions):
    def prepare_graphs(self, x_fun, x_par, y1_fun, y1_par, y2_fun, y2_par, show_plots=True):
        """
        Prepare a plot of the x and y data for a given set of parameters.

        :param x_fun: The dataframe containing the x data.
        :param x_par: The index of the column in the x dataframe containing the desired x data.
        :param y1_fun: The dataframe containing the y1 data (training function).
        :param y1_par: The index of the column in the y1 dataframe containing the desired y1 data.
        :param y2_fun: The dataframe containing the y2 data (ideal function).
        :param y2_par: The index of the column in the y2 dataframe containing the desired y2 data.
        :param show_plots: A boolean indicating whether or not to show the resulting plot. Defaults to True.
        :return: The resulting plot.
    
        """
        x = x_fun.iloc[:, x_par]    
        y1 = y1_fun.iloc[:, y1_par]     
        y2 = y2_fun.iloc[:, y2_par]     

        # create ColumnDataSource object for the data
        source = ColumnDataSource(data=dict(x=x, y1=y1, y2=y2))

        # create the figure
        p = figure(title="Train Function vs. Ideal Function", x_axis_label="x", y_axis_label="y")

        # add the two lines to the figure
        p.line(x='x', y='y1', line_width=2, color='red', legend_label="Train function", source=source)
        p.line(x='x', y='y2', line_width=2, color='blue', legend_label="Ideal function", source=source)

        # add legend to the plot
        p.legend.location = "top_left"

        # show the plot
        if show_plots:
            output_file("graph.html")
            show(p)
        else:
            return p
    

class SqliteDb(MatchFunctions):
    """
    Load data into Sqlite database
    """

    def db_and_table_creation(self, dataframe, db_name, table_name):
        """
        Parameters:

        dataframe : pandas.DataFrame
        The dataframe to be loaded into the database.

        db_name : str
        The name of the database file to be created.

        table_name : str
        The name of the table to be created in the database.

        Returns:

        None. The method saves the database file into the same folder as the project.
        Raises:

        Any exceptions raised by the underlying SQLite or SQLAlchemy libraries are passed through to the user.
        
        """
        try:
            engine = create_engine(f"sqlite:///{db_name}.db", echo=True) 
            sqlite_connection = engine.connect()  
            for i in range(len(dataframes)): 
                dataframez = dataframe[i]
                dataframez.to_sql(table_name[i], sqlite_connection, if_exists="fail")  
            sqlite_connection.close()   
        except Exception:
            exception_type, exception_value, exception_traceback = sys.exc_info()  
            print(exception_type, exception_value, exception_traceback)   



train = pd.read_csv("../train.csv")
ideal = pd.read_csv("../ideal.csv")
test = pd.read_csv("../test.csv")


df = MatchFunctions().Identify_best_fits(train, ideal)
print(df)

test = test.sort_values(by=["x"], ascending=True)   
test = test.reset_index()   
test = test.drop(columns=["index"])     

ideals = []
for i in range(0, 4):
    ideals.append(ideal[["x", f"y{str(df.iloc[i, 0])}"]])

for i in ideals:
    test = test.merge(i, on="x", how="left")

test = MatchFunctions().search_ideal_match(test)

for i in range(0, 4):
    test["Ideal index"] = test["Ideal index"].replace([i], str(f"y{df.iloc[i, 0]}"))

test = test.drop(columns=["y6", "y21", "y7", "y5"])
print(test)

train = train.rename(columns={"y1": "Y1 (training func)", "y2": "Y2 (training func)",
                              "y3": "Y3 (training func)", "y4": "Y4 (training func)"})

for col in ideal.columns:     
    if len(col) > 1:    
        ideal = ideal.rename(columns={col: f"{col} (ideal func)"})

test = test.rename(columns={"x": "X (test func)",
                            "y": "Y (test func)",
                            "Deviation": "Delta Y (test func)",
                            "Ideal index": "No. of ideal func"})

# Load data to sqlite
dbs = SqliteDb()
dataframes = [train, ideal, test]
table_names = ["train_table", "ideal_table", "test_table"]
dbs.db_and_table_creation(dataframes, "Hirak_Python_assignment", table_names)

plt.clf()
x = train.iloc[:, 0]
for i in range(1, len(train.columns)):
    plt.plot(x, train.iloc[:, i], c="g", label=f"Train function y{i}")
    plt.legend(loc=3)
    plt.show()
    plt.clf()

plt.clf()
x = ideal.iloc[:, 0]
for i in range(1, len(ideal.columns)):
    plt.plot(x, ideal.iloc[:, i], c="#FF4500", label=f"Ideal function y{i}")
    plt.legend(loc=3)
    plt.show()
    plt.clf()

plt.clf()
x = train.iloc[:, 0]
for i in range(0, df.index[-1] + 1):
    y = df.iloc[i, 0]  # get ideal y column number (18, 3, 30, 23)
    plt.plot(x, ideal.iloc[:, y], c="#FF4500", label=f"Ideal function y{y}")
    plt.legend(loc=3)
    plt.show()
    plt.clf()

plt.clf() 
plt.scatter(test.iloc[:, 0], test.iloc[:, 1])  # select x and y values
plt.show()
plt.clf()
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 6], label="Ideal - Y6", color="#33e467")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 21], label="Ideal - Y21", color="#3369e4")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 7], label="Ideal - Y7", color="#d133e4")
plt.plot(ideal.iloc[:, 0], ideal.iloc[:, 5], label="Ideal - Y5", color="#d6370d")
plt.xlabel("x")
plt.xlabel("y")
plt.legend()
plt.show()
x_fun = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
y1_fun = pd.DataFrame({'y1': [1, 3, 5, 7, 9]})
y2_fun = pd.DataFrame({'y2': [2, 4, 6, 8, 10]})
bokeh_graphs = BokehGraphs()
bokeh_graphs.prepare_graphs(x_fun, 0, y1_fun, 0, y2_fun, 0, show_plots=True)
