# Text Summarization Using Poincare Compression and Sequence Routing

# Overview 
This project to perform text summarization on Amazon review dataset using Poincare compression and sequence routing. The proposed model has been Parameter Efficient Fine tuned using LoRA.
The ROUGE scores were competetive in a low resource setting.

# Dataset
Amazon review dataset consists of millions of reviews. We sampled 130 reviews for the training and 73 for testing. We have annotated sentences as relevant or nor for the summary using semantic similarity and Rouge L scores.

# Usage 
Navigate to the project directory: <br>
python train.py

# Evaluation
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

# Dependecies
 <ul>
  <li>Python3.x</li>
  <li>Other dependecies are in the requirements.txt</li>
</ul> 

# Acknolwedgements
 <ul>
  <li><a href="https://arxiv.org/pdf/2312.00752"> Mamba State Space Model </a> </li> 
   <li><a href=" https://jmcauley.ucsd.edu/data/amazon/index_2014.html">Amazon review dataset </a> </li>
</ul> 



