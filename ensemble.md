Mean test-set accuracy: 0.9632  
Min test-set accuracy:  0.9600  
Max test-set accuracy:  0.9649  
  
  
pred_labels.shape  
(5, 10000, 10)  
  
pred_labels[0][0]  
array([  1.10406143e-06,   6.86313939e-08,   1.50151482e-06,
         2.55680352e-05,   1.46738643e-07,   8.31612454e-07,
         4.30165729e-11,   9.99926209e-01,   4.74850452e-07,
         4.41607954e-05])

  
ensemble_pred_labels = np.mean(pred_labels, axis=0)  
ensemble_pred_labels.shape  
(10000, 10)
  
  
  
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)  
ensemble_cls_pred.shape  
(10000,)
  
  

np.sum(ensemble_correct)  
9691
  
  

np.sum(best_net_correct)  
9649
  
  
    
conclusion :    
 - after softmax value(probability)

   소프트맥스한결과에서(확률값)평균값을구하여argmax



([ 0.964 ,  0.9649,  0.9632,  0.96  ,  0.9639])

9691
9649
