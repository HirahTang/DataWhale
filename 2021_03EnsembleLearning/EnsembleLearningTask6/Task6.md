## Confusion Matrix sets

## 关于分类模型的筛选：通过调整模型threshold，绘制多组confusion matrices 来找到最优化的threshold


```python
def threshold(data, value):
    data = (data >= value)
    return data

def cf_matrix_plot(cf_matrix, threshold, ax):
    
    group_names = ['TN','FP','FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    sns.heatmap(cf_matrix, ax = ax,  annot=labels, fmt='', cmap='Blues', annot_kws={"fontsize":15})
    ax.set_title('Threshold @ {}'.format(round(threshold, 2)))
#    plt.show()
    
    
def confusion_matrix_plot(eval_dt, threshold_v, ax):
    eval_dt = eval_dt.apply(lambda x: threshold(x, threshold_v), axis = 1)
    eval_dt = eval_dt.astype(int)
    tn, fp, fn, tp = confusion_matrix(eval_dt['true'], eval_dt['pred']).ravel()
    cf_matrix = confusion_matrix(eval_dt['true'], eval_dt['pred'])
    cf_matrix_plot(cf_matrix, threshold_v, ax)
    return (recall_score(eval_dt['true'], eval_dt['pred']), 
        fp / (fp + tn), precision_score(eval_dt['true'], eval_dt['pred']),
        f1_score(eval_dt['true'], eval_dt['pred']))

def cm_full(true_class, pred_class, data):
    res = data[[true_class, pred_class]]
    res = res.rename(columns={true_class: "true", pred_class: "pred"})
    fpr_l = []
    recall_l = []
    thresholds = []
    precision_l = []
    f1_l = []
    fig, axes = plt.subplots(nrows=5, ncols=4, sharex=True, sharey=True, figsize = (16, 20))

    for t_value, i in zip(np.arange(0.05, 1.05, 0.05), axes.ravel()):
#    for j in i:
        recall, fpr, precision, f1 = confusion_matrix_plot(res.copy(), t_value, i)
        fpr_l.append(fpr)
        recall_l.append(recall)
        thresholds.append(t_value)
        precision_l.append(precision)
        f1_l.append(f1)
    plt.suptitle('{}'.format(true_class), fontsize = 20)
    plt.show()
    
#    i.plot([1,2,3,4])
    plt.plot(thresholds, [round(i, 3) for i in recall_l], label = 'Recall')
    plt.plot(thresholds, [round(i, 3) for i in precision_l], label = 'Precision')
    plt.plot(thresholds, [round(i, 3) for i in f1_l], label = 'F1')
    plt.plot(thresholds, [round(i, 3) for i in fpr_l], label = 'FPR')
    plt.legend()
    plt.xlabel('Threshold values', fontsize = 15)
    plt.ylabel('Values', fontsize = 15)
    plt.title('{} for different threshold values'.format(true_class), fontsize = 17)
    plt.grid()
    plt.show()
```


```python
cm_full('Change of Use', 'ChangeofUsepred', eval_dt)
```
