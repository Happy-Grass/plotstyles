import numpy as np


class MetricsSuite:

    @staticmethod
    def input_checker(predicted, reference):
        pdims = predicted.shape
        rdims = reference.shape
        if not np.array_equal(pdims, rdims):
            message = (
                "predicted and reference field dimensions do not"
                + " match.\n"
                + "shape(predicted)= "
                + str(pdims)
                + ", "
                + "shape(reference)= "
                + str(rdims)
                + "\npredicted type: "
                + str(type(predicted))
            )
            raise ValueError(message)
        return predicted, reference

    @staticmethod
    def kling_gupta_eff09(predicted, reference, sr=1.0, salpha=1.0, sbeta=1.0):
        """
        Calculate the Kling-Gupta efficiency from 2009 paper.

        Calculates the Kling-Gupta efficiency between two variables
        predicted and reference. The kge09 is calculated using the
        formula:

        KGE09 = 1 - sqrt((cc-1)**2 + (alpha-1)**2 + (beta-1)**2)

        where:
            cc = correlation coefficient between predicted and reference;
            alpha = std(predicted) / std(reference)
            beta = sum(predicted) / sum(reference)

        where s is the predicted values, o is the reference values, and
        N is the total number of values in s & o. Note that s & o must
        have the same number of values.

        Kling-Gupta efficiency can range from -infinity to 1. An efficiency of 1 (E
        = 1) corresponds to a perfect match of model to reference data.
        Essentially, the closer the model efficiency is to 1, the more accurate the
        model is.

        The efficiency coefficient is sensitive to extreme values and might yield
        sub-optimal results when the dataset contains large outliers in it.

        Kling-Gupta efficiency can be used to quantitatively describe the accuracy
        of model outputs. This method can be used to describe the predictive
        accuracy of other models as long as there is reference data to compare the
        model results to.

        Input:
        predicted : predicted values
        reference : reference values
        sr : [optional, defaults to 1.0] scaling factor for correlation
        salpha : [optional, defaults to 1.0] scaling factor for alpha
        sbeta : [optional, defaults to 1.0] scaling factor for beta

        Output:
        kge09 : Kling-Gupta Efficiency

        References:
        Gupta, Hoshin V., Harald Kling, Koray K. Yilmaz, Guillermo F. Martinez.
        Decomposition of the mean squared error and NSE performance criteria:
        Implications for improving hydrological modelling. Journal of Hydrology,
        Volume 377, Issues 1-2, 20 October 2009, Pages 80-91. DOI:
        10.1016/j.jhydrol.2009.08.003. ISSN 0022-1694
        """
        # Check that dimensions of predicted and reference fields match
        predicted, reference = MetricsSuite.input_checker(predicted, reference)

        for name, term in [("sr", sr), ("salpha", salpha), ("sbeta", sbeta)]:
            if term > 1 or term < 0:
                raise ValueError(
                    "'{0}' must be between 0 and 1, you gave {1}".format(name, term)
                )

        std_ref = np.std(reference)
        if std_ref == 0:
            return -np.inf
        sum_ref = np.sum(reference)
        if sum_ref == 0:
            return -np.inf
        alpha = np.std(predicted) / std_ref
        beta = np.sum(predicted) / sum_ref
        cc = np.corrcoef(reference, predicted)[0, 1]

        # Calculate the kge09
        kge09 = 1.0 - np.sqrt(
            (sr * (cc - 1.0)) ** 2
            + (salpha * (alpha - 1.0)) ** 2
            + (sbeta * (beta - 1.0)) ** 2
        )

        return kge09

    @staticmethod
    def kling_gupta_eff12(predicted, reference, sr=1.0, sgamma=1.0, sbeta=1.0):
        """
        Calculate the Kling-Gupta efficiency from 2012 paper.

        Calculates the Kling-Gupta efficiency between two variables
        predicted and reference. The kge12 is calculated using the
        formula:

        kge12 = 1 - sqrt((sr*(cc-1))**2 +
                        (sgamma*(gamma-1))**2 +
                        (sbeta*(beta-1))**2
                        )

        where:
            cc = correlation coefficient between predicted and reference;
            gamma = coefficient_of_variance(predicted) /
                    coefficient_of_variance(reference)
            beta = sum(predicted) / sum(reference)

        where s is the predicted values, o is the reference values, and
        N is the total number of values in s & o. Note that s & o must
        have the same number of values.

        Kling-Gupta efficiency can range from -infinity to 1. An efficiency of 1 (E
        = 1) corresponds to a perfect match of predicted to reference data.

        The efficiency coefficient is sensitive to extreme values and might yield
        sub-optimal results when the dataset contains large outliers in it.

        Kling-Gupta efficiency can be used to quantitatively describe the accuracy
        of model outputs. This method can be used to describe the predictive
        accuracy of other models as long as there is reference data to compare the
        model results to.

        Input:
        predicted : predicted values
        reference : reference values
        sr : [optional, defaults to 1.0] scaling factor for correlation
        sgamma : [optional, defaults to 1.0] scaling factor for gamma
        sbeta : [optional, defaults to 1.0] scaling factor for beta

        Output:
        kge12 : Kling-Gupta Efficiency

        References:
        Kling, H., M. Fuchs, and M. Paulin (2012), Runoff conditions in the upper
        Danube basin under an ensemble of climate change scenarios. Journal of
        Hydrology, Volumes 424-425, 6 March 2012, Pages 264-277,
        DOI:10.1016/j.jhydrol.2012.01.011
        """
        # Check that dimensions of predicted and reference fields match
        predicted, reference = MetricsSuite.input_checker(predicted, reference)

        for name, term in [("sr", sr), ("sgamma", sgamma), ("sbeta", sbeta)]:
            if term > 1 or term < 0:
                raise ValueError(
                    "'{0}' must be between 0 and 1, you gave {1}".format(name, term)
                )

        std_ref = np.std(reference)
        if std_ref == 0:
            return -np.inf
        sum_ref = np.sum(reference)
        if sum_ref == 0:
            return -np.inf

        gamma = (np.std(predicted) / np.mean(predicted)) / (
            np.std(reference) / np.mean(reference)
        )
        beta = np.sum(predicted) / np.sum(reference)
        cc = np.corrcoef(reference, predicted)[0, 1]

        # Calculate the kge12
        kge12 = 1.0 - np.sqrt(
            (sr * (cc - 1.0)) ** 2
            + (sgamma * (gamma - 1.0)) ** 2
            + (sbeta * (beta - 1.0)) ** 2
        )

        return kge12

    @staticmethod
    def nash_sutcliffe_eff(predicted, reference):
        """
        Calculate the Nash-Sutcliffe efficiency.

        Calculates the Nash-Sutcliffe efficiency between two variables
        PREDICTED and REFERENCE. The NSE is calculated using the
        formula:

        NSE = 1 - sum_(n=1)^N (p_n - r_n)^2 / sum_(n=1)^N (r_n - mean(r))^2

        where p is the predicted values, r is the reference values, and
        N is the total number of values in p & r. Note that p & r must
        have the same number of values.

        Nash-Sutcliffe efficiency can range from -infinity to 1. An efficiency of
        1 (E = 1) corresponds to a perfect match of modeled discharge to the
        observed data. An efficiency of 0 (E = 0) indicates that the model
        predictions are as accurate as the mean of the observed data, whereas an
        efficiency less than zero (E < 0) occurs when the observed mean is a better
        predictor than the model or, in other words, when the residual variance
        (described by the numerator in the expression above), is larger than the
        data variance (described by the denominator). Essentially, the closer the
        model efficiency is to 1, the more accurate the model is.

        The efficiency coefficient is sensitive to extreme values and might yield
        sub-optimal results when the dataset contains large outliers in it.

        Nash-Sutcliffe efficiency can be used to quantitatively describe the
        accuracy of model outputs other than discharge. This method can be used to
        describe the predictive accuracy of other models as long as there is
        observed data to compare the model results to. For example, Nash-Sutcliffe
        efficiency has been reported in scientific literature for model simulations
        of discharge, and water quality constituents such as sediment, nitrogen,
        and phosphorus loading.

        Input:
        PREDICTED : predicted values
        REFERENCE : reference values

        Output:
        NSE : Nash-Sutcliffe Efficiency

        """
        predicted, reference = MetricsSuite.input_checker(predicted, reference)

        # Calculate the NSE
        nse = 1 - (
            np.sum((predicted - reference) ** 2)
            / np.sum((reference - np.mean(reference)) ** 2)
        )

        return nse

    @staticmethod
    def bias(predicted, reference):
        """
        Calculate the bias (B) between two variables PREDICTED and
        REFERENCE (E'). The latter is calculated using the formula:

        B = mean(p) - mean(r)

        where p is the predicted values, and r is the reference values.
        Note that p & r must have the same number of values.

        Input:
        PREDICTED : predicted field
        REFERENCE : reference field

        Output:
        B : bias between predicted and reference
        """
        MetricsSuite.input_checker(predicted, reference)
        # Calculate means
        b = np.mean(predicted) - np.mean(reference)
        return b

    @staticmethod
    def bias_percent(predicted, reference):
        """
        Calculate the percentage bias (B) between two variables PREDICTED and
        REFERENCE (E'). The latter is calculated using the formula:

        BP = 100*(mean(p) - mean(r))/mean(r)

        where p is the predicted values, and r is the reference values.
        Note that p & r must have the same number of values.

        Input:
        PREDICTED : predicted field
        REFERENCE : reference field

        Output:
        B : bias between predicted and reference
        """
        MetricsSuite.input_checker(predicted, reference)
        # Calculate means
        model = np.mean(predicted)
        ref = np.mean(reference)
        if ref != 0.0:
            bp = 100 * abs((model - ref) / ref)
        else:
            bp = np.NAN
        return bp

    def brier_score(forecast,observed):
        '''
        Calculate Brier score (BS) between two variables

        Calculates the Brier score (BS), a measure of the mean-square error of
        probability forecasts for a dichotomous (two-category) event, such as
        the occurrence/non-occurrence of precipitation. The score is calculated
        using the formula:

        BS = sum_(n=1)^N (f_n - o_n)^2/N

        where f is the forecast probabilities, o is the observed probabilities
        (0 or 1), and N is the total number of values in f & o. Note that f & o
        must have the same number of values, and those values must be in the
        range [0,1].

        Input:
        FORECAST : forecast probabilities
        OBSERVED : observed probabilities

        Output:
        BS : Brier score

        Reference:
        Glenn W. Brier, 1950: Verification of forecasts expressed in terms
        of probabilities. Mon. We. Rev., 78, 1-23.

        D. S. Wilks, 1995: Statistical Methods in the Atmospheric Sciences.
        Cambridge Press. 547 pp.
        '''

        forecast, observed = MetricsSuite.input_checker(forecast, observed)

        # Check for valid values
        index = np.where(np.logical_or(forecast < 0, forecast > 1))
        if sum(index) > 0:
            raise ValueError('Forecast has values outside interval [0,1].')
        index = np.where(np.logical_and(observed != 0, observed != 1))
        if sum(index) > 0:
            raise ValueError('Observed has values not equal to 0 or 1.')

        # Calculate score
        bs = np.sum(np.square(forecast - observed))/len(forecast)

        return bs

    def centered_rms_dev(predicted,reference):
        '''
        Calculates the centered root-mean-square (RMS) difference between
        two variables PREDICTED and REFERENCE (E'). The latter is calculated
        using the formula:

        (E')^2 = sum_(n=1)^N [(p_n - mean(p))(r_n - mean(r))]^2/N

        where p is the predicted values, r is the reference values, and
        N is the total number of values in p & r. Note that p & r must
        have the same number of values.

        Input:
        PREDICTED : predicted field
        REFERENCE : reference field

        Output:
        CRMSDIFF : centered root-mean-square (RMS) difference (E')^2
        '''
        predicted, reference = MetricsSuite.input_checker(predicted, reference)

        # Calculate means
        pmean = np.mean(predicted)
        rmean = np.mean(reference)

        # Calculate (E')^2
        crmsd = np.square((predicted - pmean) - (reference - rmean))
        crmsd = np.sum(crmsd)/predicted.size
        crmsd = np.sqrt(crmsd)

        return crmsd

    def rmsd(predicted,reference):
        '''
        Calculate root-mean-square deviation (RMSD) between two variables

        Calculates the root-mean-square deviation between two variables
        PREDICTED and REFERENCE. The RMSD is calculated using the
        formula:

        RMSD^2 = sum_(n=1)^N [(p_n - r_n)^2]/N

        where p is the predicted values, r is the reference values, and
        N is the total number of values in p & r. Note that p & r must
        have the same number of values.

        Input:
        PREDICTED : predicted values
        REFERENCE : reference values

        Output:
        R : root-mean-square deviation (RMSD)
        '''
        predicted, reference = MetricsSuite.input_checker(predicted, reference)

        # Calculate the RMSE
        r = np.sqrt(np.sum(np.square(predicted - reference))/len(predicted))
        return r 
