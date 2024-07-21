# Text Summarization Using Poincare Compression and Sequence Routing in Low Resource Setting

# <h2>Overview</h2>
This project to perform text summarization on Amazon review dataset using Poincare compression and sequence routing. The proposed model has been Parameter Efficient Fine tuned using LoRA.
The ROUGE scores were competetive in a low resource setting.

![Model](https://github.com/user-attachments/assets/cb6504c2-f2e7-466f-b7a1-a60207bb5de3)


# <h2>Dataset</h2>
Amazon review dataset consists of millions of reviews. We sampled 130 reviews for the training and 73 for testing. We have annotated sentences as relevant or nor for the summary using semantic similarity and Rouge L scores.

# <h2>Usage</h2> 
Navigate to the project directory: <br>
python train.py

# <h2>Evaluation</h2>
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

# <h2>Dependecies</h2>
 <ul>
  <li>Python3.x</li>
  <li>Other dependecies are in the requirements.txt</li>
</ul> 

# <h2>Acknolwedgements</h2>
 <ul>
  <li><a href="https://arxiv.org/pdf/2312.00752"> Mamba State Space Model </a> </li> 
   <li><a href=" https://jmcauley.ucsd.edu/data/amazon/index_2014.html">Amazon review dataset </a> </li>
</ul> 



