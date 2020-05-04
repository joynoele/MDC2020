using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MDC2020
{
    public class RateOfReturnSimulation
    {
        [LoadColumn(0)]
        public int Run;

        [LoadColumn(1)]
        public string Security;

        [LoadColumn(2)]
        public float Year;

        [LoadColumn(3)]
        public float Price;

        [LoadColumn(4)]
        public float Delta;

        [LoadColumn(5)]
        public Boolean IsSplit;

        [LoadColumn(6)]
        public Boolean IsBust;

        [LoadColumn(7)]
        public float Yield;

        [LoadColumn(8)]
        [ColumnName("Label")]
        public float AvgRateOfReturn;
    }

    public class RateOfReturnPrediction
    {
        [ColumnName("Score")]
        public float AvgRateOfReturn;
    }
}
