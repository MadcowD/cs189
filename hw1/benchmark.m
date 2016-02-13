function [err_rate, wrong]=benchmark(pred_labels,true_labels)
err_rate=sum(pred_labels~=true_labels)./length(true_labels);
wrong=find(pred_labels~=true_labels);
