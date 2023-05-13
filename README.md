# Runtime Monitoring
Most models are trained to predict certain task, however, in production most of them suffer from unfamiliar inputs.
<br />
Therefor, monitoring input data and detect wether the model familiar with these inputs, is a must.
<br />
In our approach, we will be using BDD (Binary Decision Diagram) to alert the user if unfamiliar pattern extracted from the last hidden layer is fed to the model.
<br />
Furthermore, we experiment with PCA to try to improve our results.
<br />

# Steps to achive final results:
1. Construct a model with high accuracy (~95%)
2. Extract last hidden layer
3. Build BDD from only correclty classified train data
4. Monitor the results
5. Apply PCA then compare

<br />