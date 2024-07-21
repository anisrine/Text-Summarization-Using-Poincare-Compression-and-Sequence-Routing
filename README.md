<h2>Text Summarization Using Poincare Compression and Sequence Routing in Low Resource Setting </h2> 

 <h3>Overview</h3>
This project to perform text summarization on Amazon review dataset using Poincare compression and sequence routing. The proposed model has been Parameter Efficient Fine tuned using LoRA.
The ROUGE scores were competetive in a low resource setting.


![Model](https://github.com/user-attachments/assets/495a24d4-7981-4ca1-8ad8-79170661bb09)



 <h3>Dataset</h3>
Amazon review dataset consists of millions of reviews. We sampled 130 reviews for the training and 73 for testing. We have annotated sentences as relevant or nor for the summary using semantic similarity and Rouge L scores.

<h3>Usage</h3> 
Navigate to the project directory: <br>
python train.py
 <h2>Evaluation</h2>
This is the best obtained ROUGE scores :
 <table>
  <tr>
    <th>Rouge1</th>
    <th>Rouge2</th>
    <th>RougeL</th>
  </tr>
  <tr>
    <td>0.214</td>
    <td>0.12</td>
    <td>0.2</td>
  </tr>

</table> 

<h3>Dependecies</h3>
 <ul>
  <li>Python3.x</li>
  <li>Other dependecies are in the requirements.txt</li>
</ul> 

<h3>Acknolwedgements</h3>
 <ul>
  <li><a href="https://arxiv.org/pdf/2312.00752"> Mamba State Space Model </a> </li> 
   <li><a href=" https://jmcauley.ucsd.edu/data/amazon/index_2014.html">Amazon review dataset </a> </li>
</ul> 



