import numpy as np
import tensorflow as tf
import os
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from nilearn import image

batch_size=16
z_dim =100
g_dim=16
mtm =0.9
epsilon = 1e-5

def input_image():
    data_dir = './coursework_myversion/data/reg/'
    patients =  os.listdir(data_dir)
    #print(len(patients))
    all_scans = {}
    i=0
    #bar = progressbar.ProgressBar()
    for patient in patients:
        if patient.endswith('.nii'):
            full_name = os.path.join(data_dir, patient)
            slices = nib.load(full_name)
            #NEED TO RESIZE
            targ_affine = np.eye(4)*1.15 #heuristic
            targ_affine[:3,3] = slices.affine[:3,3]/1.1
            targ_shape = nib.Nifti1Image(np.zeros((64,64,64)), targ_affine)
            reshaped=image.resample_to_img(slices,targ_shape)
            data = reshaped.get_data()
            '''with open('processed.csv','a') as f:
                for row in data:
                    np.savetxt(f, row,delimiter=',',fmt='%d',footer='====')'''
            midpoint = int(np.floor(data.shape[2]/2))
            midslice=data[:,:,midpoint]
            #plt.imshow(midslice, cmap=plt.cm.bone)
            #plt.show()
            midsection=data[:,:,midpoint-8:midpoint+8]
            if i%15==0:
                print("Processing scans:{}%".format(100*i/312, dtype=float))
            all_scans[i]=midsection
            i+=1
            #bar.update(100*i/int(len(patients)))
    print(np.shape(all_scans[200]))
    return all_scans

def plot(samples):
    fig = plt.figure(figsize=(64,64))
    gs = gridspec.GridSpec(4,4)
    gs.update(wspace=0.05, hspace=0.05)
    for i in range(np.shape(samples)[2]):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(samples[:,:,i].reshape(64,64), cmap=plt.cm.bone)
    return fig

def conv3d(input_, output_dim, scop, k_h=5, k_w=5, k_d=5):
    with tf.variable_scope('conv{}'.format(scop)) as scope:
        w=tf.get_variable('w{}'.format(scop), [k_h, k_w, k_d,input_.get_shape()[-1],output_dim],
            initializer=tf.truncated_normal_initializer(), dtype=tf.float32)
        conv=tf.nn.conv3d(input_, w, strides=[1,2,2,2,1], padding='SAME')
        b=tf.get_variable('b',[output_dim],initializer=tf.constant_initializer(0.0))
        conv=tf.reshape(tf.nn.bias_add(conv,b),conv.get_shape())
    return conv, w, b

def deconv3d(input_, output_size, scop, k_h=5, k_w=5, k_d=5):
    with tf.variable_scope('deconv{}'.format(scop)) as scope:
        w=tf.get_variable('w{}'.format(scop), [k_h, k_w, k_d, output_size[-1], input_.get_shape()[-1] ],
            initializer=tf.random_normal_initializer(), dtype=tf.float32)
        conv=tf.nn.conv3d_transpose(input_, w, output_shape=output_size, strides=[1,2,2,2,1], padding='SAME')
        b=tf.get_variable('b',[output_size[-1]],initializer=tf.constant_initializer(0.0))
        deconv=tf.reshape(tf.nn.bias_add(conv,b),conv.get_shape())
    return deconv, w, b

def linear(input_, output_dim, scop):
    with tf.variable_scope('lin_{}'.format(scop)) as scope:
        shape = input_.get_shape().as_list()
        matrix = tf.get_variable("Matrix", [shape[1], output_dim], tf.float32,
            tf.random_normal_initializer())
        bias = tf.get_variable("bias",[output_dim],initializer=tf.constant_initializer(0.0))
    return tf.matmul(input_, matrix)+bias, matrix, bias

def batch_norm(input_, scop,train=True):
    return tf.contrib.layers.batch_norm(input_, decay=mtm, updates_collections=None,
    epsilon=epsilon,scale=True,is_training=train,scope="batch_norm_{}".format(scop))

def generator(z, img_w, img_h, img_d):
    z0,w0,b0=linear(z, img_h*img_w*img_d*g_dim*8/(16*16*16), 'gen') #None * 144*9*9*1*8 = 5832
    h0=tf.reshape(z0, [-1, img_w/16, img_h/16, img_d/16, g_dim*8]) #9*9*1*8*14 = 5832
    h0 = tf.nn.relu(batch_norm(h0, "g_h0"))

    c1, w1, b1 = deconv3d(h0, [batch_size, img_w/8, img_h/8, img_d/8, g_dim*4],'_1' ) #10*18*18*18*576 = 33592320
    h1= tf.nn.relu(batch_norm(c1,"g_h1"))

    c2, w2, b2 = deconv3d(h1, [batch_size, img_w/4, img_h/4, img_d/4, g_dim*2],'_2' )
    h2 =tf.nn.relu(batch_norm(c2,"g_h2"))

    c3, w3, b3 = deconv3d(h2, [batch_size, img_w/2, img_h/2, img_d/2, g_dim],'_3' )
    h3 =tf.nn.relu(batch_norm(c3,"g_h3"))

    c4, w4, b4 = deconv3d(h3, [batch_size, img_w, img_h, img_d, 1],'_4' )
    h4=tf.nn.tanh(c4)

    print("shape h4",tf.shape(h4))
    theta_g = [w1, w2, w3, w4, b1, b2, b3, b4]
    return h4, theta_g

def discriminator(x,img_w, img_h, img_d, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    x=tf.reshape(x, shape=[batch_size,img_w, img_h, img_d,1])
    x=tf.cast(x, tf.float32)

    c1, w1, b1=conv3d(x ,16, '_1')
    h1=tf.nn.relu(batch_norm(c1,"d_h1"))

    c2, w2, b2=conv3d(h1,32, '_2')
    h2=tf.nn.relu(batch_norm(c2,"d_h2"))

    c3, w3, b3=conv3d(h2,64, '_3')
    h3=tf.nn.relu(batch_norm(c3,"d_h3"))
    h3=tf.reshape(h3, [batch_size,-1])

    l4, w4, b4=linear(h3,1, 'dis')
    h4=tf.nn.sigmoid(l4)
    theta_d=[w1, w2, w3, w4, b1, b2, b3, b4]
    print(l4)
    return h4,l4, theta_d

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


all_scans = input_image()
print(type(all_scans))
single_scan = all_scans[200]
img_w, img_h, img_d = np.shape(single_scan)[0], np.shape(single_scan)[1], np.shape(single_scan)[2]
batch_scan = np.array([all_scans[k] for k in range(1,batch_size+1)])
#print(type(batch_scan))
#print(np.shape(batch_scan))
x = tf.placeholder(tf.float32, shape=[None]+[img_h,img_w,img_d])
z = tf.placeholder(tf.float32, shape=[None,z_dim])
g_z, theta_g = generator(z, img_w, img_h, img_d)
d_real, d_logit_real, theta_d = discriminator(batch_scan,img_w, img_h, img_d)
d_fake, d_logit_fake, theta_d = discriminator(g_z,img_w, img_h, img_d, reuse=True)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logit_real, tf.ones_like(d_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logit_fake, tf.zeros_like(d_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(d_logit_fake, tf.ones_like(d_logit_fake)))
'''
D_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
G_loss = -tf.reduce_mean(tf.log(d_fake))
'''
d_optim = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_d)
g_optim = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_g)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
for it in range(1000000):
    print it
    if it % 10 == 0:
        print("here: ",i)
        samples = sess.run(g_z, feed_dict={z:sample_Z(batch_size, z_dim)})
        i += 1
        print("herre")
        print(np.shape(samples[1]))
        fig = plot(samples[1])
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        batch_images = np.array([all_scans[k] for k in range(i,i*batch_size+1)])
        #batch_images = batch_images.reshape([batch_size, img_h, img_w, img_d])

    #batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
    _, D_loss_curr = sess.run([d_optim, D_loss], feed_dict={x: batch_images, z:sample_Z(batch_size, z_dim)})
    _, G_loss_curr = sess.run([g_optim, G_loss], feed_dict={z: sample_Z(batch_size, z_dim)})

    if it % 10 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()

processed.close()
