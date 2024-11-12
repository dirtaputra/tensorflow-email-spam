// index.js
const tf = require('@tensorflow/tfjs');
const { data, labels } = require('./data');

// Konversi data dan label menjadi tensor
const trainingData = tf.tensor2d(data, [data.length, data[0].length]);
const outputData = tf.tensor2d(labels, [labels.length, 1]);

// Membangun model jaringan saraf sederhana
const model = tf.sequential();
model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [data[0].length] }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Kompilasi model
model.compile({
  optimizer: tf.train.adam(),
  loss: 'binaryCrossentropy',
  metrics: ['accuracy']
});

// Melatih model
(async function trainModel() {
  console.log('Training model...');
  await model.fit(trainingData, outputData, {
    epochs: 100,
    batchSize: 2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
      }
    }
  });

  // Data baru untuk diprediksi
  const newEmail = tf.tensor2d([[3, 0, 1]]); // Contoh email baru dengan kata "promo" 3 kali, "diskon" 0 kali, dan "free" 1 kali
  const prediction = model.predict(newEmail);
  prediction.print(); // Tampilkan hasil prediksi
})();
