import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_softmax_cross_entropy_multilabel(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    preds1,preds2,preds3 = tf.split(preds, [4,4,4], 1)
    labels1,labels2,labels3 = tf.split(labels, [4,4,4], 1)
    loss1 = tf.nn.softmax_cross_entropy_with_logits(logits=preds1, labels=labels1)
    loss2 = tf.nn.softmax_cross_entropy_with_logits(logits=preds2, labels=labels2)
    loss3 = tf.nn.softmax_cross_entropy_with_logits(logits=preds3, labels=labels3)
    loss = loss1 + loss2 + loss3
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
    
def masked_accuracy_multilabel(preds, labels, mask):
    """Accuracy with masking."""
    preds1,preds2,preds3 = tf.split(preds, [4,4,4], 1)
    labels1,labels2,labels3 = tf.split(labels, [4,4,4], 1)
    correct_prediction1 = tf.equal(tf.argmax(preds1, 1), tf.argmax(labels1, 1))
    correct_prediction2 = tf.equal(tf.argmax(preds2, 1), tf.argmax(labels2, 1))
    correct_prediction3 = tf.equal(tf.argmax(preds3, 1), tf.argmax(labels3, 1))
    accuracy_all = tf.cast(correct_prediction1, tf.float32) + tf.cast(correct_prediction2, tf.float32) + tf.cast(correct_prediction3, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all = accuracy_all * mask / 3
    return tf.reduce_mean(accuracy_all)
