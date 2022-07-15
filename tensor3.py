import tensorflow as tf
tensor=tf.constant([[10,7],[2,6]])
#Arithmetic Operation
tensor+10
tensor*2
tensor-1
tensor/2
tf.multiply(tensor,4)
tf.add(tensor,3)

tensor2=tf.constant([[3,4],[5,1]])
#tensor multiplication
tf.multiply(tensor,tensor2)
#tensor addition
tf.add(tensor,tensor2)
#matrix multiplication of tensors
tf.matmul(tensor,tensor2)
matrix1=tf.constant([[1,2,5],[7,2,1],[3,3,3]])
matrix1
matrix2=tf.constant([[3,5],[6,7],[1,8]])
matrix2
tf.matmul(matrix1,matrix2)
#python matrix multiplication
matrix1 @ matrix2
mat=tf.constant([[1,2,3],[4,3,2]])
mat
#tensor reshape
tf.reshape(mat,shape=(3,2))
#tensor transpose
tf.transpose(mat)
#tensor dot product/matmul
tf.tensordot(matrix1,matrix2,axes=1)
