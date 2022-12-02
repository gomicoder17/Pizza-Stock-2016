import json
import pandas as pd
import xgboost as xgb


def extract():
    """Load the data from the CSV file into the program."""
    ingredients = pd.read_csv("data/ingredients_per_day_2016.csv", index_col=0)
    ingredients.index = pd.to_datetime(ingredients.index)
    ingredients = ingredients.sort_index()
    dates = pd.date_range(ingredients.index[0], ingredients.index[-1])
    ingredients = ingredients.reindex(dates, method="ffill")
    ingredients = ingredients.fillna(0)
    return ingredients


def load_model():
    """Load the model from the file"""
    model = xgb.XGBRegressor()
    model.load_model("model/ingredients.model")
    model_info = json.load(open("model/info.json"))
    return model, model_info


def get_days():
    """Returns current day and the next monday to predict the ingredients of that week"""
    today = pd.Timestamp.now().normalize()
    today = pd.Timestamp(year=2016, month=today.month, day=today.day)
    monday = pd.Timestamp(year=2016, month=today.month, day=today.day)
    while monday.dayofweek != 0:
        monday += pd.Timedelta(days=1)
    return today, monday


def mean_ingredients_predict(
    day,
    dataframe=None,
    model=None,
    model_info=None,
    previous_days=3,
    previous_weeks=2,
    verbose=False,
):
    """
    Returns the mean of the ingredients of the previous days and the same day of the previous weeks to
    use as input for the model
    """
    if verbose:
        print("Mean ingredients to train Day: ", day.date())
    start_day = day - pd.Timedelta(days=previous_days)
    last_day = day - pd.Timedelta(days=1)
    pdays = pd.date_range(start_day, last_day)
    """Append previous weeks"""
    for w in range(previous_weeks, 0, -1):
        date = day - pd.Timedelta(weeks=w)
        pdays = pdays.insert(previous_weeks - w, date)
    """Fill missing data"""
    for day in pdays:
        if day not in dataframe.index:
            if verbose:
                print("Missing previous day:", day.date())
            delta = day - dataframe.index[0]
            if verbose:
                print("Attempting to predict it")
            if delta.days < previous_weeks * 7:
                if verbose:
                    print("Not enough data to predict!")
                    print("Returning mean of previous and next days")
                m = 1
                try:
                    return (
                        dataframe.loc[day - pd.Timedelta(days=m)]
                        + dataframe.loc[day + pd.Timedelta(days=m)]
                    ) / (2 * m)
                except:
                    if verbose:
                        print("To enought data to calculate mean")
                        print("Returning mean of all data")
                    return dataframe.mean()
            dataframe.loc[day] = predict(day, dataframe, model, model_info)
    return dataframe.loc[pdays].mean()


def filter_dataframe(dataframe, day):
    """Crops dataframe of future data which in reality would not be known"""
    return dataframe.loc[:day].copy()


def predict(day, dataframe, model, model_info):
    """
    Predict the ingredients for the given day
    Inputs:
        day: datetime object
        verbose: bool
        dataframe: dataframe with the ingredients
    Outputs:
        ingredients: dataframe with the ingredients
    """

    previous_days = model_info["previous_days"]
    previous_weeks = model_info["previous_weeks"]

    mean = mean_ingredients_predict(
        day,
        dataframe=dataframe,
        model=model,
        model_info=model_info,
        previous_days=previous_days,
        previous_weeks=previous_weeks,
        verbose=False,
    )

    dayofweek = day.dayofweek
    month = day.month
    X = pd.DataFrame([mean], columns=mean.index)
    X["dayofweek"] = dayofweek
    X["month"] = month
    pred = model.predict(X).tolist()[0]
    return pd.Series(pred, index=mean.index)


def predict_week(ti):
    """Returns the prediction of the ingredients for the next week, predicting each day and adding
    up the ingredients plus a margin to overestimate the ingredients"""
    dataframe = ti.xcom_pull(task_ids="extract")
    today, monday = ti.xcom_pull(task_ids="get_days")
    model, model_info = ti.xcom_pull(task_ids="load_model")

    dataframe = filter_dataframe(dataframe, today)

    alpha = model_info["alpha"]
    maes = pd.Series(model_info["maes"]) * alpha

    weekly_ingredients = predict(monday, dataframe, model, model_info) + maes

    for i in range(1, 7):
        day = monday + pd.Timedelta(days=i)
        ingredients = predict(day, dataframe, model, model_info)
        weekly_ingredients += ingredients + maes

    weekly_ingredients = weekly_ingredients.apply(lambda x: int(x + 0.5))

    """Write the prediction to csv"""
    weekly_ingredients.to_csv(
        f"io/predictions_for_week_{monday.date()}.csv", header=False
    )


if __name__ == "__main__":
    predict_week()
