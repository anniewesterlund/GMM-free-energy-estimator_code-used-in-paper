function p = GMM_log_likelihood(amplitudes,projections)

p = sum(log(amplitudes*projections./sum(amplitudes)));

end
