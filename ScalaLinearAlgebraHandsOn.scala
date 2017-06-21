// Databricks notebook source
// MAGIC %md 
// MAGIC # Linear Algebra and Distributed Machine Learning in Scala using Breeze and MLlib
// MAGIC The objective of this hands-on is to introduce you to the use of:
// MAGIC - the *Scala* programming language,
// MAGIC - linear algebra operations using *Breeze*,
// MAGIC - the MLlib distributed machine learning library of Spark.
// MAGIC 
// MAGIC First, I will introduce you to the Scala [*Breeze*](https://github.com/scalanlp/breeze/) library. Then, I will exemplify two basic statistical methods, principal component analysis and linear regression analysis using both Breeze and [MLlib](https://spark.apache.org/docs/latest/mllib-guide.html). Throughout the examples, you will recognize that a good command of Breeze commands is very helpful even if we directly use the MLlib library. This hands-on will finish with an introduction to parameter tuning using MLib's tuning package. 
// MAGIC 
// MAGIC This hands-on is based on code examples from DataBricks notebooks (especially the *Apache Spark on Databricks for Data Scientists* notebook) and the documentation of the Breeze library. Some sections mainly reproduce, combine and slightly modify what was already presented there, other sections are more innovative in terms of implementing something new (Implementing Standard Linear Regression Analysis with Breeze,
// MAGIC Implementing Ridge Regression with Breeze, Using Heuristics for Parameter Tuning).
// MAGIC 
// MAGIC ### Introduction to the Databricks Spark Environment
// MAGIC At the very beginning of your Spark session on the Databricks cluster, you have to start a cluster, open a new Scala notebook, and associate this notebook to the running cluster.
// MAGIC 
// MAGIC Please start this hands-on by reading the very basic [*Getting started*](https://docs.databricks.com/user-guide/getting-started.html) tutorial.
// MAGIC 
// MAGIC Then, go through the more detailed [*A Gentle Introduction to Apache Spark on Databricks*](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055043/484361/latest.html) tutorial.
// MAGIC 
// MAGIC You should also read the [*Notebook tutorial*](https://docs.databricks.com/user-guide/notebooks/index.html) in order to get started with the notebook format.
// MAGIC You might skip this tutorial, if you are already familiar with jupyter notebooks. Databricks notebooks are associated with a given programming language.
// MAGIC However, they may contain cells with code from other languages (Scala, Python, R, SQL, bash) and additional markdown or html cells. 
// MAGIC Cells with non-default languages have to start with a special macro such as e.g. "%sql" for SQL code or "%md" for markdown. 
// MAGIC 
// MAGIC In the following, I will assume that you are familiar with the content of these introductory tutorials.
// MAGIC 
// MAGIC ## Linear Algebra in Scala
// MAGIC The following introduction to *Breeze* is partly based on the [Breeze Tutorial](https://github.com/scalanlp/breeze/wiki/Quickstart) and the [Breeze Linear-Algebra-Cheat-Sheet](https://github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet).
// MAGIC 
// MAGIC ### Working with Vectors
// MAGIC #### Creating Vectors
// MAGIC Let's start with a few examples of creating dense and sparse vectors. Working with sparse vectors is especially important in big data situations where it is common to have many cells with missing or zero values. Using sparse data structures can significantly speed up computations and reduce memory and CPU/GPU footprint. 

// COMMAND ----------

import breeze.linalg._
import breeze.numerics._

//create a dense vector 
val v = DenseVector(1.0,2.0,3.0,4.0,5.0)

//create a vector of zeros
val zeros = DenseVector.zeros[Double](5)

//create a vector of ones
val ones = DenseVector.ones[Double](5)

//create a vector with a particular number (5.0 repeated 10 times)
val fives = DenseVector.fill(10){5.0}

//create a vector of random numbers (uniform random numbers between 0 and 1)
val random = DenseVector.rand(5)

//create a sparse vector
val sparse =  SparseVector(0,1.0,0,3.0,0)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Unary Operators 
// MAGIC Some important unary operators, for a full list checkout the  [Breeze Linear-Algebra-Cheat-Sheet](https://github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet).

// COMMAND ----------

//get the number of elements of a vector
println("vector length: "+v.length)

//access elements of a vector
println("The first element of v: "+v(0)) 

//negative indexes have the effect of accessing element in inverse order, starting at the end of the vector
println("The last element of v: "+v(-1)) 

//create a range of numbers
val range = 1 to 3
//and extract the vector elements within this index range
println("The elements "+range+" of v: "+v(range)) 

//calculate the sum of all elements (result is a scalar)
val sumOf_v = sum(v)

//calculate the cumulative sum (result is a vector)
val cumSum_v = accumulate(v)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Binary Operators
// MAGIC Some important binary operators (involving two vectors), for a full list checkout the  [Breeze Linear-Algebra-Cheat-Sheet](https://github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet).

// COMMAND ----------

//add two vectors elementwise
println(v+zeros)

//add a dense and a sparse vector elementwise (the result is a DENSE vector)
println(v+sparse)

//elementwise comparison (returns boolean vector)
println(v :< sparse)

//the dot product (i.e. inner product)
println("Dot product: ")
println(v dot sparse)

// COMMAND ----------

// MAGIC %md
// MAGIC #### Functional Programming
// MAGIC Scala is a programming language that combines the object oriented and functional paradigm. 
// MAGIC We can implement pure functional programes with Scala which makes it ideal for the map reduce programming model.
// MAGIC The functional paradigm does not allow for side-effects and it does not allow variables, instead only constants should be used.
// MAGIC In Scala, we declare a constant using the "val" keyword. (Variables are declared with "var", we will only use them in the last section of this hands-on.)

// COMMAND ----------

//Perform a list of transformations on the elements of a vector 
//In Spark this represents a stage which is a pipeline of transformations which can be performed in parallel.
//The actual computation is triggered by the final action using println().
v.map(v => exp(v)).map(v=>2*v).map(v=>println(v))

//Map-reduce example:
//Map operations: Do transformations of vector elements in parallel.
//Reduce: Combine the final values of all vector elements to one value using the reduce function (here sum).
val x = DenseVector(1.0,2.0,3.0,3.0,4.0,4.0)
x.map(x=>exp(x)).map(x=>ceil(x)).map(x=>exp(x)).reduce( _ + _ )

// COMMAND ----------

// MAGIC %md
// MAGIC ### Working with Matrices
// MAGIC #### Creating Matrices
// MAGIC As for vectors, we can create dense and sparse matrices. There are also special constructors for special matrix types, such as e.g. the identity matrix or a matrix of zeros. For a full list checkout the  [Breeze Linear-Algebra-Cheat-Sheet](https://github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet).

// COMMAND ----------

//Building a dense matrix inline
val m1 = DenseMatrix((1.1,2.2,0.4), (3.9,4.333,0.3242),(-0.43443,9.4242,-21324.3))

//Building a dense matrix inline
val m3 = DenseMatrix((1.1,2.2,0.4,0.8), (3.9,4.333,0.3242,0.71253),(-0.43443,9.4242,-21324.3,0.21738))

//A dense matrix of zeros with 3 rows and 3 columns
val zeros = DenseMatrix.zeros[Double](3,3)

//An identity matrix with dimension 5
val identity = DenseMatrix.eye[Double](5)

//A matrix with random elements between 0 and 1 with 3 rows and 5 columns
val random = DenseMatrix.rand(10,10)

//Creating a sparse matrix with 10 rows and 9 columns and two nonzero elements
val builder = new CSCMatrix.Builder[Double](rows=10, cols=9)
builder.add(3,4, 1.11)
builder.add(1,2, 0.001)
val sparseM = builder.result()

// COMMAND ----------

// MAGIC %md 
// MAGIC #### Unary Operations
// MAGIC A list of useful operations on a single matrix, for a full list checkout the  [Breeze Linear-Algebra-Cheat-Sheet](https://github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet).

// COMMAND ----------

//Get a specific column of a matrix (indexing starts at zero!)
val firstColumn = m1(::,0)

//Get a specific row of a matrix (transpose the matrix first)
val firstRow = m1.t(::,0)

//Transpose 
println("Matrix m1:")
println(m1)
println("")
println("transpose of m1:")
println(m1.t)
println("")

//Determinant 	
println("The determinant of the matrix \"random\":")
println(det(random))
println("")

//Inverse
println("The inverse of the matrix \"random\":")
val random_inv = inv(random)
println(random_inv)
println("")

//Moore-Penrose Pseudoinverse (for non-square matrices)
//This algorithm generalises the inverse to non-square matrices.
//It also makes sense to use it for square matrices because it is more robust numerically than the inverse function.
println("The Moore-Penrose pseudoinverse of the matrix \"m3\":")
val m3_mp_inv = pinv(m3)
println(m3_mp_inv)
println("")

//Rank
println("The rank of matrix \"m3\":")
println(rank(m3))
println("")

//Singular Value Decomposition
val svd.SVD(u,s,v) = svd(random)
println("The singular value decomposition of matrix \"random\":")
println("The left singular vectors:")
println(u)
println("")

println("The right singular vectors:")
println(v)
println("")

println("The singular values:")
println(s)
println("")

// COMMAND ----------

// MAGIC %md 
// MAGIC #### Binary Operations
// MAGIC A small list of operations on two matrices.

// COMMAND ----------

//matrix multiplication of two matrices with the same dimensions
val m4 = m1 * m3

//The result of multiplying a square matrix with the identity matrix (of the same dimensionality) is again the same matrix.
println(m1 == m1 * DenseMatrix.eye[Double](3))

//Mutiplying a matrix with a vector is a vector.
val m5 = m1 * DenseVector.ones[Double](3)

//elementwise matrix operation + 
m1 + zeros == m1

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Broadcasting
// MAGIC Sometimes we have to express that we want to apply the same function to all columns or all rows of a matrix.
// MAGIC If you are familiar with *R* think of the apply() function. In Scala broadcasting is done using the "\*" sign (compare the  [Breeze Tutorial](https://github.com/scalanlp/breeze/wiki/Quickstart)). 

// COMMAND ----------

import breeze.stats._

println(m1)

//Example adapted from Breeze tutorial: https://github.com/scalanlp/breeze/wiki/Quickstart
//print the column means of matrix m1
println(mean(m1(::,*)))

//print the row means of matrix m1
println(mean(m1(*,::)))

//more functions

println("The sum of columns:")
println(sum(m1(::,*)))

println("The variance of columns:")
println(variance(m1(::,*)))

println("The standard deviation of columns:")
//Here I first have to first transpose the vector and transform it into an array before I can use map reduce.
variance(m1(::,*)).t.toArray.map(x=>sqrt(x)).map(x=>println(x))

println("The median of columns:")
println(median(m1(::,*)))

// COMMAND ----------

// MAGIC %md
// MAGIC Broadcasting can be used in combination with [universal functions](https://github.com/scalanlp/breeze/wiki/Universal-Functions). It is even possible to define new universal functions which can then be applied to all the rows or columns of a matrix, this however involves a deeper understanding of how scala traits works (compare the section *Using UFuncs in generic functions* on the [universal functions](https://github.com/scalanlp/breeze/wiki/Universal-Functions) web page). 
// MAGIC 
// MAGIC Now that we have learned how to implement linear algebra operations in Scala, let's start to implement statistical algorithms.

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Principal Component Analysis based on a Covariance Matrix
// MAGIC In this small exercise, we will use the vector and matrix operations we learned before to get started with singular value decomposition (SVD) and multivariate statistics. 
// MAGIC We will not use external data sources and instead create a random correlation matrix. 
// MAGIC A correlation matrix is a standardized covariance matrix with ones in the diagonal and correlations in the off-diagonal entries. 
// MAGIC The correlation matrix of a data set with p variables has p rows and p columns. 
// MAGIC The diagonal describes the standardized variance, the off-diagonal entries descibe the correlations between variables. 
// MAGIC Based on the results of singular value decomposition, we can compute the eigenvalues and the eigenvectors in the space defined by the inputs (i.e. variables) which has dimensionality p. 
// MAGIC Given that we do not have a data matrix, we can only calculate the projection of the variables onto these eigenvectors. 
// MAGIC This projection of the variables is simply the product between the singular values (square root of eigenvalues) and the eigenvectors in the p-dimensional space defined by the variables. 
// MAGIC We can not compute the principal components (the projection of the observations onto the eigenvectors in Rp), because we do not have a data matrix.
// MAGIC In the next example, I will run a principal component analysis based on a real data set.

// COMMAND ----------

import breeze.math._
import breeze.linalg.NumericOps 
import breeze.linalg.operators 
import breeze.stats.distributions._

//The number of observations (cases/ instances):
val N = 100

//The number of features (inputs/ variables):
val p = 10

//Create a variance-covariance matrix describing the multivariate covariance structure:
val A: DenseMatrix[Double] = lowerTriangular(DenseMatrix.rand(p,p))

//Create a symmetric matrix by mirroring the lower triangular matrix.
val B = A + A.t

val C = DenseMatrix.ones[Double](p,p)
//The off-diagonal elements are all between 0 and 1, so we have to substract a matrix with 0.5 and double the values afterwards.

val D = 2.0 * (B - 0.5 * C)

//set diagonal to 1 (correlation matrix with equal variance of inputs)
diag(D) := 1.0
println(D)

//Based on the correlation matrix we are already able to run a principal component analysis based on Singular Value Decomposition:
//https://github.com/scalanlp/breeze/blob/master/math/src/main/scala/breeze/linalg/functions/svd.scala
val svd.SVD(u,s,v) = svd(D)
println("The singular value decomposition of matrix \"random\":")
println("The left singular vectors:")
println(u)
println("These are the eigenvectors in Rn!")

//Let's have a look at the dimensionality of this matrix:
//Get the first column of matrix u (the first eigenvector):
val u1 = u(::, 1)
println("Number of columns: "+u1.length)
//get the first row of matrix u:
val u_1 = u.t(::,1)
println("Number of rows: "+u_1.length)
//Conclusion: The eigenvectors in Rn should have dimensionality N!
//This is not the case because we base our analysis directly on the covariance matrix instead of working with the original data matrix.

println("The right singular vectors:")
println(v)
println("These are the eigenvectors in Rp!")
//get the dominant eigenvector
val v1 = v(::, 1)
println("Number of columns: "+v1.length)
val v_1 = v.t(::,1)
println("Number of rows: "+v_1.length)
println("")

println("The singular values:")
println(s)
//The singular values are the square root of the eigenvalues.
println("The eigenvalues:")
s.map(v=>v*v).map(v=>println(v))
println("")

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Implementing Principal Component Analysis with MLib
// MAGIC This time, we will work with a publicly available data set descibing tax revenue in the United States. 
// MAGIC This data set called *SOI Tax Stats - Individual Income Tax Statistics - ZIP Code Data (SOI)* is available in the distributed file system of databricks so we can directly use it.
// MAGIC A description of the data set is available [here](https://www.irs.gov/uac/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi). 
// MAGIC The code for loading the data set and visualizing it using SQL commands is based on the [*A Gentle Introduction to Apache Spark on Databricks*](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055043/484361/latest.html) tutorial.
// MAGIC We will select some numeric variables of the data set, assemble them into a data matrix, and run a principal component analysis of the data matrix. Let's first import the data set into a Spark DataFrame:

// COMMAND ----------

//High level SQL data set:
// in Apache Spark 2.0
val taxes2013 = spark.read
   .option("header", "true")
   .csv("dbfs:/databricks-datasets/data.gov/irs_zip_code_data/data-001/2013_soi_zipcode_agi.csv")

//create a temporary view
taxes2013.createOrReplaceTempView("taxes2013")

// COMMAND ----------

// MAGIC %md
// MAGIC Now we have created a Spark DataFrame. In Spark there are three main data abstractions: RDD (Resilient Distributed Dataset), DataFrame, and Dataset.  
// MAGIC RDDs are the lowest level API to a distributed "sequence of data objects". The DataFrame API is comparable to data sets in *R* or *Python* pandas.
// MAGIC The Dataset API is the newest interface to distributed data. According to Databricks, it "can be considered a combination of DataFrames and RDDs. It provides the typed interface that is available in RDDs while providing a lot of conveniences of DataFrames. It will be the core abstraction going forward."  Source: [*A Gentle Introduction to Apache Spark on Databricks*](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055043/484361/latest.html)
// MAGIC 
// MAGIC Based on the DataFrame abstraction, we can now use SQL commands to display and manipulate the newly created tabel:

// COMMAND ----------

// MAGIC %sql show tables

// COMMAND ----------

display(taxes2013)

// COMMAND ----------

// MAGIC %md
// MAGIC We can also directly print the data base schema:

// COMMAND ----------

taxes2013.printSchema()

// COMMAND ----------

// MAGIC %md
// MAGIC We conclude that all except the first four variables contain floating point values which we can use for our data matrix.
// MAGIC 
// MAGIC Now we will read in the data as RDD which is the basic data abstraction of Spark.

// COMMAND ----------

import org.apache.spark.mllib.linalg.{Vector, Vectors}

//Low level RDD:
val rawdata = sc.textFile("dbfs:/databricks-datasets/data.gov/irs_zip_code_data/data-001/2013_soi_zipcode_agi.csv")

//remove the header
val header = rawdata.first()
println("File header: " + header)
val data = rawdata.filter(_(0) != header(0))

//count the number of lines
val N = data.count()

val parsedData = data.map { line =>
        //For each line, split the String at the separator, which is "," as we saw in the header.
		val parts = line.split(',')
        //For each line, only select values starting at index 5.
		val values = parts.slice(5, parts.length)
        Vectors.dense(values.map(_.toDouble))
}.cache()

import org.apache.spark.mllib.stat.Statistics

//Calculate the correlation matrix from the data matrix
val R = Statistics.corr(parsedData, "pearson")

// COMMAND ----------

// MAGIC %md
// MAGIC Now we will create a *distributed* matrix of type *RowMatrix* in Spark.
// MAGIC This type of distributed matrix makes sense if the rows do not have any order as is the case in standard data sets, where rows represent observations without any inherent ordering.

// COMMAND ----------

//Compare: 
//https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.distributed.RowMatrix

val rows: RDD[Vector] = parsedData // an RDD of local vectors
// Create a RowMatrix from an RDD[Vector].
val mat: RowMatrix = new RowMatrix(rows)

// Get the dimensionality of the new distributed matrix:
//number of observations n
val n = mat.numRows()
//number of variables p
val p = mat.numCols()

// COMMAND ----------

//Source: https://spark.apache.org/docs/2.1.1/mllib-dimensionality-reduction.html
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors

// Compute only the top k singular values and corresponding singular vectors.
val k = 10
val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU = true)
val U: RowMatrix = svd.U  // The U factor is a RowMatrix.
val U_num_rows = U.numRows()
val U_num_cols = U.numCols()
println("The matrix U of left singular vectors is a distributed matrix of type RowMatrix with "+U_num_rows+" rows and "+U_num_cols+" columns.")
println("These are the eigenvectors in Rn!")
println("Summary statistics for these eigenvectors:" + U.computeColumnSummaryStatistics())

val V: Matrix = svd.V  // The V factor is a local dense matrix.
val V_num_rows = V.numRows
val V_num_cols = V.numCols
println("The matrix V of right singular vectors is a local dense matrix with "+V_num_rows+" rows and "+V_num_cols+" columns.")
println("These are the eigenvectors in Rp!")

val s: Vector = svd.s  // The singular values are stored in a local dense vector.
println("The singular values:")
println(s)
//The singular values are the square root of the eigenvalues.
println("The eigenvalues:")
s.toArray.map(v=>v*v).map(v=>println(v))
println("")


// COMMAND ----------

// MAGIC %md
// MAGIC #### Conclusion: 
// MAGIC The eigenvalues are huge, let's standardize the eigenvalues by dividing by their sum.

// COMMAND ----------

val sum_eigenvalues = s.toArray.map(v=>v*v).reduce(_+_)
s.toArray.map(v=>v*v).map(v=>println(v/sum_eigenvalues))

// COMMAND ----------

// MAGIC %md
// MAGIC Now we can clearly see that the first principal component explains 97% of the total variance (approximated by the variance of the first 10 components).
// MAGIC The problem here clearly is that the variables are measured in different units, therefore one variable dominates all the others. 
// MAGIC Let's first scale and standardize our data before we rerun the SVD.

// COMMAND ----------

//https://stackoverflow.com/questions/36736411/how-to-normalize-or-standardize-the-data-having-multiple-columns-variables-in-sp#36738288
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors

// Creating a Scaler model that standardizes with both mean and SD
val scaler = new StandardScaler(withMean = true, withStd = true).fit(parsedData)

// Scale features using the scaler model
val scaledFeatures = scaler.transform(parsedData)


// COMMAND ----------

// MAGIC %md
// MAGIC Now that we have standardized the input data, we will again transform the RDD into a distributed matrix and run SVD.

// COMMAND ----------

val rows: RDD[Vector] = scaledFeatures // an RDD of local vectors
// Create a RowMatrix from an RDD[Vector].
val mat: RowMatrix = new RowMatrix(rows)

// Get the dimensionality of the new distributed matrix:
//number of observations n
val n = mat.numRows()
//number of variables p
val p = mat.numCols()

// Compute only the top k singular values and corresponding singular vectors.
val k = 20
val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU = true)
val U: RowMatrix = svd.U  // The U factor is a RowMatrix.
val U_num_rows = U.numRows()
val U_num_cols = U.numCols()
println("The matrix U of left singular vectors is a distributed matrix of type RowMatrix with "+U_num_rows+" rows and "+U_num_cols+" columns.")
println("These are the eigenvectors in Rn!")
println("Summary statistics for these eigenvectors:" + U.computeColumnSummaryStatistics())

val V: Matrix = svd.V  // The V factor is a local dense matrix.
val V_num_rows = V.numRows
val V_num_cols = V.numCols
println("The matrix V of right singular vectors is a local dense matrix with "+V_num_rows+" rows and "+V_num_cols+" columns.")
println("These are the eigenvectors in Rp!")

val s: Vector = svd.s  // The singular values are stored in a local dense vector.
println("The singular values:")
println(s)
//The singular values are the square root of the eigenvalues.
println("The eigenvalues:")
s.toArray.map(v=>v*v).map(v=>println(v))

// COMMAND ----------

val sum_eigenvalues = s.toArray.map(v=>v*v).reduce(_+_)
val explained_variance = DenseVector(s.toArray.map(v=>v*v).map(v=>v/sum_eigenvalues))
println("The cumulative explained variance of the eigenvalues:")
println(accumulate(explained_variance))

// COMMAND ----------

// MAGIC %md 
// MAGIC It makes sense to work with the first 3 principal components, because they cumulatively explain 93% of total inertia (i.e. variance). Let' calculate the principal components (projections of the variables) as well as the projections of the individuals (the scores).

// COMMAND ----------

//Source: https://github.com/apache/spark/blob/master/examples/src/main/scala/org/apache/spark/examples/mllib/PCAOnRowMatrixExample.scala
//Calculate the first three principal components:
val pc: Matrix = mat.computePrincipalComponents(3)
println("The matrix of principal components has "+pc.numRows+" rows and "+pc.numCols+" columns.")

// Project the individuals into the linear space spanned by the top 3 principal components:
val projected: RowMatrix = mat.multiply(pc)
println("The projection matrix of the individuals has "+projected.numRows+" rows and "+projected.numCols+" columns.")

val collect = projected.rows.collect()
println("Projected Row Matrix of principal component:")
collect.foreach { vector => println(vector) }

// COMMAND ----------

// MAGIC %md 
// MAGIC ## Linear regression
// MAGIC ### Implementing Standard Linear Regression Analysis with Breeze

// COMMAND ----------

// MAGIC %md
// MAGIC Linear regression has long been a cornerstone of statistics and although there are many limitations it still is a very attractive method in big data situations because as a parametric model the complexity of the model only depends on the number of input (variables) and not on the number of observations (cases) and the coefficients can be estimated very efficiently using standard matrix algebra (closed form solution).
// MAGIC This is true for both standard linear regression (no regularization) and ridge regression (L2 norm regularization). 
// MAGIC 
// MAGIC In order to be independent of external data sets and fully understand our data generating model, let's first create a dummy data set with uncorrelated features. 
// MAGIC Then, we will create random coefficients and calculate the output (response) y as a linear combination of the x.
// MAGIC In the next step, we will use standard matrix algebra to solve for the betas (the regression coefficients).
// MAGIC Note that this is an artificial example because there is no noise. We will introduce noise in the next code block.

// COMMAND ----------

//Compare: http://www.statsblogs.com/2013/12/30/brief-introduction-to-scala-and-breeze-for-statistical-computing-2/
import breeze.stats.distributions._
import breeze.stats.DescriptiveStats._

//standard Gaussian distribution for the means of inputs 
val mean_dist = Gaussian(0.0,1.0)

//lognormal distribution for the variance of inputs 
val var_dist = LogNormal(5.0,3.0)

//The number of observations (cases/ instances):
val N = 100
val max_index_train = (N/2)-1
val start_index_test = max_index_train +1

//The number of features (inputs/ variables):
val p = 10

//empty data matrix, with p columns
val X = DenseMatrix.zeros[Double](N,p)
//The first half of the observations is stored as training set.
val X_train = DenseMatrix.zeros[Double]((N/2), p)
//The second half of the observations is stored as test set.
val X_test = DenseMatrix.zeros[Double]((N/2), p)

//create all inputs (predictors)
for(i <- 0 to (p-1)){  
  val mean = mean_dist.draw()
  val sd = var_dist.draw()
  val dist = Gaussian(mean,sd)
  val x = dist.sample(N)
  //matrix assignment to column
  X (::,i) := DenseVector(x.toArray)
  X_train(::,i) := DenseVector(x.toArray)(0 to max_index_train)
  X_test(::,i) := DenseVector(x.toArray)(start_index_test to (N-1))
}

//calculate y based on inputs and random coefficients
val coefs = mean_dist.sample(p)
val c = DenseVector(coefs.toArray)
val y = X * c
val y_train =y(0 to max_index_train)
val y_test = y(start_index_test to (N-1))
//estimate the coefficients using least squares:
val b = pinv(X.t * X) * X.t * y 

println("Sum of squared differences between the true and the estimated coefficients:")
println((c-b).map(x=>x*x).reduce(_+_))

// COMMAND ----------

// MAGIC %md
// MAGIC Conclusion: The estimated betas are very close to the true coefficients!
// MAGIC Now let's introduce some gaussian noise.

// COMMAND ----------

//create a vector of random numbers (uniform random numbers between 0 and 1)
val noise = DenseVector(mean_dist.sample(N).toArray)

//add the noise vector with mean zero and variance 1
val y = X * c + noise
//estimate the coefficients using least squares:
val b = pinv(X.t * X) * X.t * y 

println("Sum of squared differences between the true and the estimated coefficients:")
println((c-b).map(x=>x*x).reduce(_+_))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Conclusion
// MAGIC The estimate is still pretty good but less accurate than before.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Implementing Ridge Regression with Breeze
// MAGIC Standard regression analysis runs into problems when dealing with many correlated inputs (co-linearity of predictors).
// MAGIC Different strategies have been employed to counteract this problem:
// MAGIC a) best subset selection: select the optimal set of inputs, avoid inputs with strong correlations among each other.
// MAGIC b) principal component regresion: use orthogonal principal components instead of the origial variables.
// MAGIC c) regularized regression: use shrinkage of the regression coefficients / penalizing the coefficients using either L1 (Lasso), L2 (Ridge), or intermediate (elastic net) norm.
// MAGIC One of the key practical advantages of regularized regression approaches is that the data analysts is not forced to run a feature selection or a selection of significant principal components beforehand.
// MAGIC Feature selection has the big disadvantage of increased variance! Running feature selection for different samples from the same data source, we will select different varying subsets of features. 
// MAGIC Regularization on the other hand forces us to use cross-validation in order to tune the regularization paramter lambda.
// MAGIC Here, I will show how we can implement one special case of regularized regression (ridge regression). Later I will show how to implement regularized regression for both ridgre regression, Lasso, and elastic net using MLlib.  

// COMMAND ----------

//Let's just use an arbitrary value for the regularization parameter lambda for now:
val lambda = 0.1
val I = DenseMatrix.eye[Double](p)
val b = pinv(X.t * X + lambda * I) * X.t * y 

println("Sum of squared differences between the true and the estimated coefficients:")
println((c-b).map(x=>x*x).reduce(_+_))

// COMMAND ----------

// MAGIC %md
// MAGIC Now let's calculate the predictions of the model and calculate the residuals and the residual sum of squares.

// COMMAND ----------

val y_hat = X * b
val residuals = y - y_hat
val sum_of_squares = residuals.map(x=>x*x).reduce( _ + _ )

// COMMAND ----------

// MAGIC %md
// MAGIC Let's optimize the value of the regularization parameter by iterating over a range of values.
// MAGIC We use half of the observations to estimate the coefficients and we use the other half of the observations to assess the accuracy of the model for unseen data.
// MAGIC In practice we would have to implement a more advanced cross-validation scheme such as 5-fold cross-validation.

// COMMAND ----------

for(x <- 1 to 10){ 
  val lambda = 1.0 / ( x * x )
  val I = DenseMatrix.eye[Double](p)
  val b = pinv(X_train.t * X_train + lambda * I) * X_train.t * y_train 
  val y_hat = X_test * b
  val residuals = y_test - y_hat
  val sum_of_squares = residuals.map(x=>x*x).reduce( _ + _ )
  println("lambda: "+lambda+" residual sum of squares: "+sum_of_squares)
}

// COMMAND ----------

// MAGIC %md 
// MAGIC #####Conclusions: 
// MAGIC For this data set with uncorrelated inputs, we can choose a very low value of lambda. 
// MAGIC In other words regularization is not neccesary. As a general conclusion, although it is possible to implement ridge regression with Breeze, performing parameter tuning is rather clumsy.
// MAGIC It is also much less trivial to implement Lasso or elastic net regression. Although, be aware that there is already a Lasso implementation in the Breeze stats/regression package. We will therefore turn to the MLlib library.

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Implementing linear regression analysis with MLib
// MAGIC #### Some data processing
// MAGIC In order to prepare the regression analysis with MLlib, I will now create a new data set from an existing Databricks data set. 
// MAGIC The code of this section is taken from the [*A Gentle Introduction to Apache Spark on Databricks*](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055043/484361/latest.html) tutorial. 
// MAGIC 
// MAGIC First, we will create a new SQL table from the already existing *taxes2013* table: 

// COMMAND ----------

// MAGIC %sql 
// MAGIC DROP TABLE IF EXISTS cleaned_taxes;
// MAGIC 
// MAGIC CREATE TABLE cleaned_taxes AS
// MAGIC SELECT state, int(zipcode / 10) as zipcode, 
// MAGIC   int(mars1) as single_returns, 
// MAGIC   int(mars2) as joint_returns, 
// MAGIC   int(numdep) as numdep, 
// MAGIC   double(A02650) as total_income_amount,
// MAGIC   double(A00300) as taxable_interest_amount,
// MAGIC   double(a01000) as net_capital_gains,
// MAGIC   double(a00900) as biz_net_income
// MAGIC FROM taxes2013

// COMMAND ----------

// MAGIC %md
// MAGIC Let's have a look at the existing tables after this command:

// COMMAND ----------

// MAGIC %sql show tables

// COMMAND ----------

// MAGIC %md
// MAGIC We can display the first 1000 cases (observations) using SQL:

// COMMAND ----------

// MAGIC %sql select * FROM cleaned_taxes

// COMMAND ----------

// MAGIC %md
// MAGIC We can also use Scala commands and the *sqlContext* constant to calculate the mean net capital gain for US states:

// COMMAND ----------

val cleanedTaxes = sqlContext.table("cleaned_taxes")
display(cleanedTaxes.groupBy("state").avg("net_capital_gains"))

// COMMAND ----------

// MAGIC %md
// MAGIC Another useful command to get summary statistics:

// COMMAND ----------

display(cleanedTaxes.describe())

// COMMAND ----------

// MAGIC %md 
// MAGIC In order to have the data readily available, we will cache the new data set.
// MAGIC (For more details on caching, read the corresponding section in the [*A Gentle Introduction to Apache Spark on Databricks*](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055043/484361/latest.html) tutorial!)

// COMMAND ----------

sqlContext.cacheTable("cleaned_taxes")

// COMMAND ----------

// MAGIC %md
// MAGIC Now we have to define the columns, we want to use as features (inputs) in the regression.

// COMMAND ----------

val cleaned_taxes = sqlContext.sql("SELECT * FROM cleaned_taxes")

//replace NAs with zeros if applicable
val cleaned_taxes_no_nas = cleaned_taxes.na.fill(0)

// define columns which should be ignored because they are categorical 
val nonFeatureCols = Array("state", "zipcode")

//define columns of data set which should be used as features
val featureCols = cleaned_taxes_no_nas.columns.diff(nonFeatureCols)

// COMMAND ----------

// MAGIC %md
// MAGIC Now, all selected columns are transformed into a single vector "features" and this new vector is attached to the existing data set. 

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

//define the transformation of several input columns inta a new output column which contains the values of all selected input columns  
val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")

//create a new data set with an additional column "features"
val finalPrep = assembler.transform(cleaned_taxes_no_nas)

// COMMAND ----------

display(finalPrep)

// COMMAND ----------

// MAGIC %md 
// MAGIC We split the data into a training (80%) and a test set (20%).

// COMMAND ----------

val Array(training, test) = finalPrep.randomSplit(Array(0.8, 0.2))

// cache the data
training.cache()
test.cache() 

println("N training set: " + training.count())
println("N test set: " + test.count())

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Implementing regularized regression with MLib
// MAGIC MLlib has implemented the elastic net regularization. This means that we can choose between Lasso regression (alpha=1), ridge regression (alpha=0) and intermediate values of alpha (0<=alpha<=1).
// MAGIC 
// MAGIC The most important differences between Lasso regression and ridge regression are:
// MAGIC - Lasso regression leads to sparse solutions, i.e. some coefficients are zero. As a result, Lasso regression is implicitly performing a feature selection
// MAGIC - Ridge regression can be performed using linear algebra operations because there is a closed form solution. Lasso regression is based on optimization algorithms (quadratic programming problem).
// MAGIC If you are interested in the differences between both regularization approaches, section 3.4 of "Elements of Statistical Learning" (ISBN:978-0-387-84857-0) is a useful reference.
// MAGIC 
// MAGIC Using the elastic net (0<=alpha<=1) we can tune alpha using cross-validation and combine the advantages of Lasso and ridge regression.
// MAGIC 
// MAGIC The code in the following two cells is adapted from the [*Apache Spark on Databricks for Data Scientists (Scala)*](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/346304/2168141618055194/484361/latest.html) tutorial (section *Apache Spark MLLib*).

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression

//Define the regression model
val lrModel = new LinearRegression()
  //define the response (output) variable
  .setLabelCol("net_capital_gains")
  //define the predictors (inputs) 
  .setFeaturesCol("features")
  //define if we use: L1 (Lasso, alpha=1), L2 (Ridge regression, alpha=0) or intermediate alpha (elastic net). 
  .setElasticNetParam(0)

//Print a descriptive summary of the model and all its settings
println("A descriptive summary of the model and all its settings/ parameters:")
println(lrModel.explainParams)

// COMMAND ----------

// MAGIC %md
// MAGIC In this example we only specify the most minimal:
// MAGIC 1. The output of the model with .setLabelCol().
// MAGIC 1. The inputs to the model with .setFeaturesCol().
// MAGIC 2. The type of regularization we want to use with .setElasticNetParam().
// MAGIC 
// MAGIC For the rest of the parameters, default settings are used which may or may not make sense:
// MAGIC - By default the regularization parameter (lambda) is set to zero, so we did in fact not run a regularized regression at all! 
// MAGIC - The inputs are by default standardized but not centered and an intercept term is fitted: In some situations, we might prefer to work with the centered data and exclude the intercept term. 
// MAGIC - By default, 100 iterations are used during optimization. If the model does not converge, we can increase this number.
// MAGIC - Note that we can also supply a weight vector to the method.
// MAGIC 
// MAGIC Before we delve deeper into the details of parametrization, let's first fit the model and retrieve some metrics which will help us to assess model quality:

// COMMAND ----------

import org.apache.spark.mllib.evaluation.RegressionMetrics
val lrFitted = lrModel.fit(training)

// define the test set as holdout 
val holdout = lrFitted
  .transform(test)

// RegressionMetrics
val rm = new RegressionMetrics(
  holdout.select("prediction", "net_capital_gains").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("MSE: " + rm.meanSquaredError)
println("MAE: " + rm.meanAbsoluteError)
println("RMSE Squared: " + rm.rootMeanSquaredError)
println("R Squared: " + rm.r2)
println("Explained Variance: " + rm.explainedVariance + "\n")

// COMMAND ----------

// MAGIC %md
// MAGIC #### Conclusion
// MAGIC The model fit is very good. The coefficient of determination is close to 100%. This must be due to the fact that we included some predictors which are very correlated to the response.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Parameter Tuning with MLlib
// MAGIC ### Parameter Tuning based on Grid Search 
// MAGIC In any real data analysis, we have to tune the parameters of our machine learning method to the data set using cross-validation. 
// MAGIC For regularized regression, the key hyper-parameter is the regularization parameter lambda.  
// MAGIC Parameter tuning is especially complex and time consuming if we work with models that have several interacting hyper-parameters and if our data analysis involves a pipeline of methods. In our case, we might tune both alpha and lambda at the same time.
// MAGIC The MLlib library offers a convenient framework for hyper-parameter tuning which is based on parameter grid evaluation.
// MAGIC Although there are more efficient ways of parameter tuning (the number of grid points explodes with increasing number of parameters),
// MAGIC we will for now use this convenient parameter tuning framework to tune our model. 

// COMMAND ----------

//Compare: https://spark.apache.org/docs/latest/ml-tuning.html
//Code example based on "Apache Spark on Databricks for Data Scientists (Scala)" notebook
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.{Pipeline, PipelineStage}

//Here, we define all coordinates in the 2 dim parameter space that we want to sample.
val paramGrid = new ParamGridBuilder()
  .addGrid(lrModel.elasticNetParam, Array(0.0, 0.3, 0.5, 0.7, 1.0))
  .addGrid(lrModel.regParam, Array(1.0, 0.5, 0.1, 0.01))
  .build()

//Here we transform the regression model into a pipeline stage.
val steps:Array[PipelineStage] = Array(lrModel)

//We build a pipeline consisting of only one stage (our regression model).
val pipeline = new Pipeline().setStages(steps)

//We specify how and what should be cross-validated 
val cv = new CrossValidator()
  .setNumFolds(5)  // Here we set the number of folds k for k-fold cross-validation.
  .setEstimator(pipeline) // Here we set the model or sequence of models (pipeline) that we want to tune.
  .setEstimatorParamMaps(paramGrid) // Here we supply the parameter grid, i.e. the coordinates in hyper-parameter which should be sampled.
  .setEvaluator(new RegressionEvaluator().setLabelCol("net_capital_gains")) // Here we define the evaluator function for deciding which is the optimal parameter combination. This depends on the type of model used.

//Run the cross-validation over all possible parameter combinations defined in paramGrid.
val pipelineFitted = cv.fit(training)

// COMMAND ----------

// MAGIC %md
// MAGIC Then we can extract the optimal model parametrization in terms of minimal cross-validation error with the following code: 

// COMMAND ----------

import org.apache.spark.ml.param.Param

val paramMap = pipelineFitted
  .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
  .stages(0)
  .extractParamMap

//print the name and the value of the parameters only:
println("The optimal parameters are:")
paramMap.toSeq.foreach(pair => {
  println(s"${pair.param.name}: "+pair.value)
})

//extract the name of the model
var model_name = paramMap.toSeq(0).param.parent

// COMMAND ----------

// MAGIC %md
// MAGIC In most situations printing the optimal parameters will be sufficient.
// MAGIC However, we might need to store the value of the parameters in some new constant.
// MAGIC This is a bit more difficult and I will describe it below:

// COMMAND ----------

//print the complete param map
paramMap.toSeq.foreach(pair => {
  println(s"${pair.param.parent} ${pair.param.name} ${pair.param.doc}")
  println(pair.value)
})

//If we want to store the value of some of the parameters in a new constant we have to remember the name of the "parent" property
//and build a new Param object using this name.
//The signature of the constructor: new Param(parent: String, name: String, doc: String) 
var regParam = new Param(model_name, "regParam", "regularization parameter (>= 0)")
var elasticNetParam = new Param(model_name, "elasticNetParam", "the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty")

val regParam_optimal = paramMap.get(regParam)
val elasticNetParam_optimal = paramMap.get(elasticNetParam)

//The parameter values are of type optional!
//We have to check if the values is defined, then we can retrieve the value with .get
//Compare: http://alvinalexander.com/scala/using-scala-option-some-none-idiom-function-java-null
if(regParam_optimal.isDefined) println("Optimal lambda: "+regParam_optimal.get)
if(elasticNetParam_optimal.isDefined) println("Optimal alpha: "+elasticNetParam_optimal.get)


// COMMAND ----------

// MAGIC %md
// MAGIC #####Conclusion 
// MAGIC We should use a value of lambda of 0.01 (very low regularization) and we should use ridge regression (alpha=0)!
// MAGIC 
// MAGIC Let's again calculate the regression metrics of the chosen model parametrization for the test set:

// COMMAND ----------

//Code example based on "Apache Spark on Databricks for Data Scientists (Scala)" notebook.
val holdout2 = pipelineFitted.bestModel
  .transform(test)

val rm2 = new RegressionMetrics(
  holdout2.select("prediction", "net_capital_gains").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("MSE: " + rm2.meanSquaredError)
println("MAE: " + rm2.meanAbsoluteError)
println("RMSE Squared: " + rm2.rootMeanSquaredError)
println("R Squared: " + rm2.r2)
println("Explained Variance: " + rm2.explainedVariance + "\n")


// COMMAND ----------

// MAGIC %md 
// MAGIC #### Using Heuristics for Parameter Tuning 
// MAGIC In some situations, running a grid search is not enough. Sometimes we may have just *too many parameters* with *too wide ranges* of possible parameter values to sample the complete parameter space. Instead, we might have to implement our own, heuristic, optimization scheme.
// MAGIC One approach which is often used is called [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing). The basic idea is to decrease the range of possible parameter values with increasing number of iterations. In the following example, I will show how to implement such an approach in the two dimensions of our parameter space. We will first use exactly the same code as before but reorganize it. Let's reduce the number of folds k to 2 because we just want to check if everything is working. We use the pipeline and the underlying regression model from before (which is still stored in memory).

// COMMAND ----------

//A function that cross-validates our regression model 
//training: the training data set
//pipeline: the model pipeline which in our case exists only of one model
//paramGrid: the grid of hyper-parameters which are evaluated in the cross-validation
//k: number of folds used in cross-validation (at least 2, typically used values are k=5 and k=10)
def runRegression(training: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row], 
   pipeline: org.apache.spark.ml.Pipeline,
   paramGrid: Array[org.apache.spark.ml.param.ParamMap],
   k: Integer) = {
  	
   //We specify how and what should be cross-validated 
   val cv = new CrossValidator()
      .setNumFolds(k)  // Here we set the number of folds k for k-fold cross-validation to 2 to speed up things.
      .setEstimator(pipeline) // Here we set the model or sequence of models (pipeline) that we want to tune.
      .setEstimatorParamMaps(paramGrid) // Here we supply the parameter grid
      .setEvaluator(new RegressionEvaluator().setLabelCol("net_capital_gains")) // Here we define the evaluator function for deciding which is the optimal parameter combination. This depends on the type of model used.

    //Run the cross-validation over all possible parameter combinations defined in paramGrid.
    val pipelineFitted = cv.fit(training)

    //Extract the parameter map
    val paramMap = pipelineFitted
      .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(0)
      .extractParamMap
    
    //Return the optimal parameters as found by cross-validation
    (paramMap.get(regParam), paramMap.get(elasticNetParam))
}		


//Here, we define all coordinates in the 2 dim parameter space that we want to sample.
val paramGrid = new ParamGridBuilder()
  .addGrid(lrModel.elasticNetParam, Array(0.0, 0.3, 0.5, 0.7, 1.0))
  .addGrid(lrModel.regParam, Array(1.0, 0.5, 0.1, 0.01))
  .build()

val optimalParams = runRegression(training, pipeline,paramGrid,2)

// COMMAND ----------

// MAGIC %md 
// MAGIC Let's have a look at the optimal params as returned by our new function:

// COMMAND ----------

println(optimalParams)

//the first element
println(optimalParams._1)

//the second element
println(optimalParams._2)

//let's handle the optionals appropriately
if(optimalParams._1.isDefined) println("Optimal lambda: "+ optimalParams._1.get)
if(optimalParams._2.isDefined) println("Optimal alpha: "+ optimalParams._2.get)

// COMMAND ----------

// MAGIC %md
// MAGIC Now we can use our new cross-validation function to implement a simulated annealing type of optimization heuristic for the model parameters.
// MAGIC We use the distributions package of Breeze to be able to sample from probability distributions.

// COMMAND ----------

import breeze.stats.distributions._

//number of iterations of the overall simulated annealing algorithm
val maxIterationsSA = 5

//Here, we define the initial parameter grid.
//We will update this grid in each iteration.
//Note that we use var instead of val for the first time in this hands-on!
var paramGrid = new ParamGridBuilder()
  .addGrid(lrModel.elasticNetParam, Array(0.0, 0.3, 0.5, 0.7, 1.0))
  .addGrid(lrModel.regParam, Array(1.0, 0.5, 0.1, 0.01))
  .build()

var lambda_optim = 0.0 
var alpha_optim = 0.0
var step_size = 0.3
//number of samples per dimension
var grid_size = 3

for (iteration <- 1 to maxIterationsSA ){
    val optimalParams = runRegression(training, pipeline,paramGrid,2)
      
    //we remove the step size in each iteration
    step_size *= 0.7
  
    if(optimalParams._1.isDefined) {
      println("Optimal lambda: "+ optimalParams._1.get)
      lambda_optim = optimalParams._1.get
    }
    
    if(optimalParams._2.isDefined) {
      println("Optimal alpha: "+ optimalParams._2.get)
      alpha_optim = optimalParams._2.get
    }
    
    //Use a log normal distribution to create a probability distribution for the lambdas 
    var dist_lambda = LogNormal(log(lambda_optim), step_size)
  
    //updated grid for lambda by sampling grid_size values from the log normal distribution defined above
    var grid_lambda = dist_lambda.sample(grid_size).toArray
     
    //Use a Gaussian distribution to create a probability distribution for the alphas
    //Note that this is problematic, parametrizing the Beta distribution would be better.
    var dist_alpha = Gaussian(alpha_optim, sqrt(step_size))
  
    //Updated grid for alpha sampling grid_size values from the normal distribution defined above
    //We additionally have to enforce that the value is between 0 and 1.
    var grid_alpha = DenseVector(dist_alpha.sample(grid_size).toArray).map{i =>
           if (i < 0.0) 0.0 else if (i > 1.0) 1.0 else i
        }.toArray
       
    paramGrid = new ParamGridBuilder()
      .addGrid(lrModel.elasticNetParam, grid_alpha)
      .addGrid(lrModel.regParam, grid_lambda)
      .build()
}

// COMMAND ----------

// MAGIC %md 
// MAGIC ##### Conclusions:
// MAGIC Based on the *ParamBuilder* and the *CrossValidator* classes we can easily implement own own tailor-made tuning algorithm.
// MAGIC Note that the choice of step size and the grid structure should be adopted to the specific model parameter.
// MAGIC The resulting code is both concise and efficient. 

// COMMAND ----------

// MAGIC %md 
// MAGIC ### Where to go from here?
// MAGIC I recommend the book *Advanced Analytics with Spark* which is available at the UPC library in electronic form. As a UPC member you can access it [here](http://recursos.biblioteca.upc.edu/login?url=http://proquest.safaribooksonline.com./9781491912751?uicode=politicat). It is also worth exploring the Databricks notebooks. If you are searching for specific solutions for the Spark platform search on [StackOverflow](https://stackoverflow.com/) using the *apache-spark* flag. You will find Breeze related questions searching with the *scala-breeze* flag. The ultimate reference for MLlib is the [scala API documentation](https://spark.apache.org/docs/latest/api/scala/index.html#package).
