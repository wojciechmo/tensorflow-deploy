import * as tf from '@tensorflow/tfjs';
import {loadFrozenModel} from '@tensorflow/tfjs-converter';
// Load GPU binding
// import '@tensorflow/tfjs-node-gpu';
import "babel-polyfill"; // to import regeneratorRuntime for async function

async function main() 
{
	const x = tf.tensor1d([2.0, 2.0, 2.0, 2.0]);
	document.getElementById('input').innerText = x;
	
	const MODEL_URL = 'http://localhost:8081/tensorflowjs_model.pb?origin=*';
	const WEIGHTS_URL = 'http://localhost:8081/weights_manifest.json?origin=*';

	const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);
	const y = model.predict(x);

	document.getElementById('output').innerText = y;
}

main();
