#Continuation of Current Method
The approach you are taking to determining canonical correlations between frequency components of each view over each day  is interesting and you should finish the analysis. I suggest that you generate the correlation matrices and canonical correlation vectors for all subjects and then cluster them to see if common patterns over time and over subject occur. You can use K-means or hierarchical clustering on all days for all subjects and see if interesting clusters occur (clusters of pre-innocc vs post-innoc or clusters of Sx vs Asx days).
    * Might be helpful to project down using t-SNE. This could be a nice visualization for the DARPA people. There's a chance Al will be happy with that.
* Things to test:
    * All subject changes
        * Just for correlation
        * For correlation and CCA
    * New saving and loading method
        * Just correlation
        * Correlation and CCA
        * Just CCA

#Additional Experiments to Address Weaknesses of Current Method
However, your approach uses a model that may be overly restrictive for the following reasons:
    1. High correlations between the frequency components of  a pair of multiviews occur only when each view has the same pattern at roughly the same time but with possibly different time spread (the same wavelet coefficient pattern occurring at different scales). It will fail to give high correlation when a pattern in view 1 is consistently accompanied by a different pattern in view 2 or when a pattern occurring in view 1 occurs at a substantially different time in view 2.

    2. Use of the entire day for correlation averaging will be fine when view 1 and view 2 have common patterns that are localized at the same times and occur repeatedly across the entire day. If temporally localized common patterns in the views only occur once or twice per day the intervening non-common "noise" will significantly degrade the high correlation that would occur if there were no such noise.

    3. Since you are computing correlation and canonical correlation over a single day at a time it is difficult to draw conclusions about repeated patterns over different days (biochronicity) from your analysis. Indeed, a finding of similar correlation matrices or the canonical vectors (SVD) over all the days will tell us that there is a periodic occurrence of common patterns in the views but these are not necessarily the same pattern every day.

#Biochronicity-related Stuff
As we have discussed throughout the past year, our primary objective should be to explore biochronicity and its perturbation by inoculation or infection. Specifically, to find
    1. patterns in view 1 and view 2 that are possibly different but co-occur at the same time at least once a day (cross-correlation).
    2. patterns in view 1 that repeat every day in view 1, and similarly for view 2 (auto-correlation).

    * To get a quick start at the autocorrelation (item b) you can apply your existing algorithm to a multiview pair that is defined as (view1(day i), view1(day i+1)) instead of (view1(day i), view2(day i)).

    * For cross-correlation (item a) you will need to do the correlation averaging across the 8 days for a small window, e.g. one hour duration in each day, that you can slide over the 24 hour period. The magnitude of the complex wavelets will be invariant to a small shift in time (a shift that is less than the size of the support set of wavelet at a given scale). However, to make the analysis invariant to larger shifts you can detect the (k) largest wavelet coefficient magnitudes in each window and correlate the associated coefficient configurations (only relative times between the strongest wavelet coefficients will enter the analysis).

#Test Signals
To test your analysis I have prepared a pair of test signals in tab delimited files (attached). The files  (example1_TS1, example1_TS2,  example2_TS1 and example2_TS2) are two different examples of time series for two views  (TS1 and TS2 designation) for an easy case (example1) and a harder case (example2). The sampling rate for all these files is 1 sample/min.  Please use these to test your method and to explain your results to me at our next meeting.

#Random (reorganize later)
* Linear regression between accelerometer and heart rate via scatter plot
* First show 'statistical picture' (CCA heat maps), then scatter plot, then individual example, then introduce likely causal relationship between accelerometer and heart rate
* T-tests and p-values for spike in temperature vs reported symptoms
* Split APE file of Trio of Doom album into individual tracks
