from __future__ import print_function

import tensorflow as tf
import numpy as np

def get_length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length


class Model(object):
	def __init__(self, hidden_size = 75, embedding_size = 300, is_training= True):
		self.start_index = tf.placeholder(tf.int32, [None])                     #[batch_size]
		self.stop_index = tf.placeholder(tf.int32, [None]) 		                #[batch_size]
		#self.dropout_rate = tf.placeholder(tf.int32 , [1])			
		input_dim = 0
		

		with tf.name_scope("word-rep"):
			self.question_repres = tf.placeholder(tf.float32, [None, None, embedding_size])   # [batch_size, question_len, word_dim]
			self.passage_repres = tf.placeholder(tf.float32, [None, None, embedding_size])    # [batch_size, passage_len, word_dim]

			self.question_lengths = get_length(self.question_repres)				#[batch_size]
			self.passage_lengths = get_length(self.passage_repres)					#[batch_size]

			input_shape = tf.shape(self.question_repres)
			batch_size = input_shape[0]
			batch_size = tf.cast(batch_size, tf.int32)

			question_len = input_shape[1]
			input_shape = tf.shape(self.passage_repres)
			passage_len = input_shape[1]
			input_dim += input_shape[2]
	
		passage_mask = tf.sequence_mask(self.passage_lengths, passage_len, dtype=tf.float32) # [batch_size, passage_len]
		question_mask = tf.sequence_mask(self.question_lengths, question_len, dtype=tf.float32) # [batch_size, question_len]

		# - sequence length helper function
		def seq_len(seq):
			seq_bool = tf.sign(tf.abs(seq))
			return tf.reduce_sum(seq_bool, axis=-1)

		with tf.name_scope("q-p_encoder"):
			with tf.variable_scope("passage-encoder"):
				#W = tf.Variable(tf.truncated_normal(shape = [], stddev=0.05),name = "w")
				#b = tf.Variable(tf.constant(0.1, shape=[]),name="b")

				fcell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
				bcell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
				u_p,_ = tf.nn.dynamic_rnn(fcell, inputs = self.passage_repres,dtype= tf.float32, sequence_length = self.passage_lengths)
			
			with tf.variable_scope("question-encoder"):
				#W = tf.Variable(tf.truncated_normal(shape, stddev=0.05),name = "w")
				#b = tf.Variable(tf.constant(0.1, shape=[]),name="b")

				fcell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
				bcell = tf.contrib.rnn.BasicLSTMCell(hidden_size)

				u_q,_ = tf.nn.dynamic_rnn(fcell, inputs =self.question_repres,dtype= tf.float32, sequence_length = self.question_lengths)
		
		# i : batch_number , k : question_len_number
		#unstacked_u_q = tf.unstack(u_q, axis = 0,num = 10)
		#unstacked_u_p = tf.unstack(u_p,axis = 0,num = 10)
		#print(unstacked_u_q)
		with tf.name_scope("q-p_attention"):
			lstm_m_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)

			def match_attention(k, q_i, p_i, len_q_i, state, batch_tensor):#iterate k over len_p_i
				p_i_k = tf.reshape(p_i[k], [1, -1])#[1, 75]
				q_i_k = tf.slice(q_i, begin=[0,0], size=[len_q_i, hidden_size])#[len_q_i, 75]
				
				with tf.variable_scope('attn_weights'):
					w_s = tf.get_variable(shape=[hidden_size, hidden_size], name='w_s')#w_u_q
					w_t = tf.get_variable(shape=[hidden_size, hidden_size], name='w_t')#w_u_p
					w_m = tf.get_variable(shape=[hidden_size, hidden_size], name='w_m')
					w_e = tf.get_variable(shape=[hidden_size, 1], name='w_e')

				m_lstm_state = tf.reshape(state.h, [1,-1])
				sum_m = tf.matmul(q_i_k, w_s) + tf.matmul(p_i_k, w_t) + tf.matmul(m_lstm_state, w_m)
				s_k = tf.matmul(tf.tanh(sum_m), w_e)

				exps = tf.reshape(tf.exp(s_k), [len_q_i])
				alphas = exps / tf.reshape(tf.reduce_sum(exps, 0), [1])
				a_k = tf.reduce_sum(q_i* tf.reshape(alphas, [len_q_i, 1]), 0)

				c_t_Q = tf.reshape(a_k, [1,hidden_size])#c_t_Q [1, 75]
				m_k = tf.concat([p_i_k , c_t_Q], axis=1)
				with tf.variable_scope('lstm_m_step'):
					out, next_state = lstm_m_cell(inputs=m_k, state=state)
				
				batch_tensor = batch_tensor.write(k,out)
				k = tf.add(k,1)
				return k, q_i, p_i, len_q_i, next_state, batch_tensor

			def match_sentence(i, h_m_ta):
				#p_emb_i, h_emb_i = u_q[i], u_p[i]
				p_i = u_p[i]							#q_i :[question_len,input_dim] , p_i:[passage_len,input_dim]
				q_i = u_q[i]
				
				len_q_i, len_p_i = seq_len(question_mask[i]), seq_len(passage_mask[i])
				len_q_i = tf.cast(len_q_i, tf.int32)
				len_p_i = tf.cast(len_p_i, tf.int32)  
				state = lstm_m_cell.zero_state(batch_size=1, dtype=tf.float32)
				batch_tensor = tf.TensorArray(dtype=tf.float32, size=tf.cast(len_p_i, tf.int32))
				# inner loop
				k = tf.constant(0)
				c = lambda a, x, y, z, s, u: tf.less(a, tf.cast(len_p_i, tf.int32))
				b = lambda a,x,y,z,s,u  : match_attention(a,x,y,z,s,u)
				res = tf.while_loop(cond=c, body=b, 
								   loop_vars=(k, q_i, p_i, len_q_i, state, batch_tensor))
				
				temp = tf.squeeze(res[-1].stack(),axis = 1)#[len_p_i, 75]
				h_m_ta = h_m_ta.write(i, temp)
				
				i = tf.add(i,1)
				
				
				return i, h_m_ta

			with tf.variable_scope('lstm_matching'):
				h_m_ta = tf.TensorArray(dtype=tf.float32, size=batch_size)
				
				#h_m_ta = np.array([10,15,75])
				c = lambda x,y: tf.less(x, batch_size)
				b = lambda x,y: match_sentence(x,y)
				i = tf.constant(0)
				h_m_res = tf.while_loop(cond=c, body=b,
									   loop_vars = (i, h_m_ta))
				
				v_p = h_m_res[-1].stack()#[batch, len_p, 75]

		with tf.name_scope("self-matching"):
			bilstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
			
			def self_match_attention(t,p_i,len_p_i,state,batch_val):#iterate t over len_p_i
				v_p_t = tf.reshape(p_i[t],[1,-1])#[1, 75]
				v_p = p_i
				with tf.variable_scope("w"):
					w_v_p = tf.get_variable(shape = [hidden_size, hidden_size],
								  name = "w_v_p")
					w_v_p_ = tf.get_variable(shape = [hidden_size, hidden_size],
								  name = "w_v_p_")
					w_v_e = tf.get_variable(shape = [hidden_size,1],
								  name = "w_v_e")

				bilstm_state = tf.reshape(state.h,[1,-1])
				sum_m = tf.matmul(v_p,w_v_p) 
				sum_m += tf.matmul(v_p_t,w_v_p_)
				s_t = tf.matmul(tf.tanh(sum_m),w_v_e)
				exps = tf.reshape(tf.exp(s_t), [len_p_i])

				alphas = exps / tf.reshape(tf.reduce_sum(exps, 0), [1])
				a_t = tf.reduce_sum(p_i* tf.reshape(alphas, [len_p_i, 1]), 0)

				a_t = tf.reshape(a_t, [1,hidden_size])
				m_t = tf.concat([a_t, v_p_t], axis=1)
				with tf.variable_scope('lstm_m_step'):
					out, next_state = bilstm_cell(inputs=m_t, state=state)
				
				batch_val = batch_val.write(t,out)
				t = tf.add(t,1)

				return t,p_i,len_p_i,next_state,batch_val

			def self_match_sentence(i,h):
				p_i = v_p[i]#p_i: [len_p_i, 75]
				len_p_i = tf.cast(seq_len(passage_mask[i]),tf.int32)

				state = bilstm_cell.zero_state(batch_size=1, dtype=tf.float32)
				batch_val = tf.TensorArray(dtype=tf.float32, size=1)

				t = tf.constant(0)
				c = lambda a, x, y, z, s  : tf.less(a, len_p_i)
				b = lambda a, x, y, z, s  : self_match_attention(a, x, y, z, s)

				res = tf.while_loop(cond=c, body=b, loop_vars=(t,p_i,len_p_i,state,batch_val))

				tem = tf.squeeze(res[-1].stack(),axis=1)
				h = h.write(i,tem)
				i = tf.add(i,1)
				return i,h

			with tf.name_scope("lstm_self_matching"):
				h = tf.TensorArray(dtype=tf.float32, size=batch_size)
				c = lambda x,y : tf.less(x,tf.cast(batch_size, tf.int32))
				b = lambda x,y : self_match_sentence(x,y)
				i = tf.constant(0)
				res = tf.while_loop(cond=c, body=b, loop_vars=(i,h))
				h_p = res[-1].stack()
				print(h_p)
		

		with tf.variable_scope("output_layer"):
			with tf.name_scope("intial_state"):
				with tf.variable_scope("par"):
					w_v_q = tf.get_variable(shape = [hidden_size, hidden_size],name = 'w_v_q')
					w_u_q = tf.get_variable(shape = [hidden_size, hidden_size],name = 'w_u_q')
					V_r_q = tf.get_variable(shape = [15, hidden_size],name = 'V_r_q')				#15 : question_len
					e = tf.get_variable(shape = [hidden_size,1],name = 'e')
				shape_u_q = tf.shape(u_q)
				sum_m = tf.reshape(tf.matmul(tf.reshape(u_q,[-1,hidden_size]),w_u_q),shape_u_q)
				sum_m += tf.matmul(V_r_q,w_v_q)
				s = tf.matmul(tf.reshape(tf.tanh(sum_m),[-1,hidden_size]),e)  # [batch*len,1]
				exps = tf.reshape(tf.exp(s), [-1, question_len])
				alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
				initial_s = tf.reduce_sum(u_q * tf.reshape(alphas, [-1, question_len, 1]), 1)  #[batch_size,hidden_size]
				c_ = tf.zeros(shape = tf.shape(initial_s), dtype = tf.float32)
				

			with tf.name_scope("answer_recurrent_network"):
				answer_lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
				
				predictions = []
				shape_h_p = tf.shape(h_p)
				with tf.variable_scope('wi'):
					w_h_p = tf.get_variable(shape = [hidden_size,hidden_size], name = "w_h_p")
					w_h_a = tf.get_variable(shape = [hidden_size,hidden_size], name = "w_h_a")
					w_h_e = tf.get_variable(shape = [hidden_size,1], name = "w_h_e")
				for i in range(2):
					if(i==0):
						sum_m = tf.reshape(tf.reshape(tf.matmul(tf.reshape(h_p,[-1,hidden_size]),w_h_p),shape_h_p) + tf.matmul(initial_s,w_h_a),[-1,hidden_size])
						s = tf.matmul(tf.tanh(sum_m), e)
						exps = tf.reshape(tf.exp(s), [-1, passage_len])
						alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
						predictions.append(exps)
						alphas = tf.reshape(alphas, [-1,passage_len,1])
						#a_k = tf.reduce_sum(q_i* tf.reshape(alphas, [len_q_i, 1]), 0)
						input_a = tf.reduce_sum(h_p*alphas, 1)
						
						initial_s = tf.tuple([initial_s, initial_s])
					else:
						sum_m = tf.reshape(tf.reshape(tf.matmul(tf.reshape(h_p,[-1,hidden_size]),w_h_p),shape_h_p) + tf.matmul(initial_s.h,w_h_a),[-1,hidden_size])
						s = tf.matmul(tf.tanh(sum_m), e)
						exps = tf.reshape(tf.exp(s), [-1, passage_len])
						alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
						predictions.append(exps)
						alphas = tf.reshape(alphas, [-1,passage_len,1])
						#a_k = tf.reduce_sum(q_i* tf.reshape(alphas, [len_q_i, 1]), 0)
						input_a = tf.reduce_sum(h_p*alphas, 1)
					
					_, initial_s = answer_lstm(input_a, initial_s)
					

		with tf.name_scope("loss"):
			pred_start = predictions[0]   # [batch_size, passage_len]
			pred_end = predictions[1]     #  [batch_size , passage_len]

			def calc_loss(pred, ind):
				loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=ind)
				loss *= passage_mask
				return tf.reduce_sum(loss)
				'''
				loss = 0.0
				for batch in pred:
					for i,val in enumerate(batch):
						if(i==ind):
							loss+= tf.log(float(val))
						else:
							loss+= tf.log(1-float(val))
				return loss
				'''

			self.loss = calc_loss(pred_start, self.start_index) + calc_loss(pred_end, self.stop_index)

		with tf.name_scope("accuracy"):

			correct_start = tf.equal(tf.argmax(pred_start, 1, output_type=tf.int32), self.start_index)
			self.accuracy_start = tf.reduce_mean(tf.cast(correct_start, 'float'))

			correct_stop = tf.equal(tf.argmax(pred_end, 1, output_type=tf.int32), self.stop_index)
			self.accuracy_stop = tf.reduce_mean(tf.cast(correct_stop, 'float'))
