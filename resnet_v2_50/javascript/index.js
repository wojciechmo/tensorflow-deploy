
let imgElement = document.getElementById('imageSrc');
let inputElement = document.getElementById('fileInput');

inputElement.addEventListener('change', (e) => {
  imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);

async function process_classes_names(filename) 
{
	/*fetch('http://localhost:8080/imagenet_names.txt')
	.then(response => response.text())
	.then(text => console.log(text))*/
    
	var response = await fetch(filename)
	lines = await response.text()
	lines = lines.split("\n");
	
	// in imagenet_names.txt there are 1000 valid classes while model predicts also 'zero' class 	
	var classes = ['none']	
	for (i = 0; i < lines.length - 1; i++)
	{	
		line = lines[i];
		name = line.substring(7,line.length-2);
		classes.push(name.toLowerCase());
	}
	
	return classes;
}

imgElement.onload = async function() 
{
	let img = cv.imread(imgElement); //cv.IMREAD_COLOR deosn't work here  
	
	console.log('Image loaded') 
	console.log('image width: ' + img.cols + '\n' +
				'image height: ' + img.rows + '\n' +
				'image channels ' + img.channels() + '\n');
				
	cv.cvtColor(img, img, cv.COLOR_BGRA2RGB, 0);
	let dsize = new cv.Size(224,224);
	cv.resize(img,img,dsize, 0, 0, cv.INTER_AREA);

	img.convertTo(img, cv.CV_32F, alpha=1.0/255.0, beta=0.0);
	
	classes = await process_classes_names('http://localhost:8080/imagenet_names.txt')
	//console.log(classes)
	
	img_tensor = tf.tensor4d(img.data32F, [1, img.rows, img.cols, 3], dtype='float32');
	const MODEL_URL = 'http://localhost:8080/tensorflowjs_model.pb?origin=*';
	const WEIGHTS_URL = 'http://localhost:8080/weights_manifest.json?origin=*';

	const model = await tf.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
	const outputs_tensors = model.predict(img_tensor) 
	
	console.log('Classification finished.') 
	
	probs_tensor = outputs_tensors[0]
	top_k_tensor = outputs_tensors[1]
	//top_k_tensor.print()
	
	// dataSync() returns flatten TypedArray, as long as batch_size = 1 -> no reshape needed  	
	probs_data = probs_tensor.dataSync()
	top_k_data = top_k_tensor.dataSync()

	output_str = "Top 5 predicitions:\n";
	console.log("Top 5 predicitions:");
	for (i=0; i<5; i++)
	{
		const single_prediciton = "--class: " + classes[top_k_data[i]] + " --probability: " + probs_data[top_k_data[i]];
		console.log(single_prediciton);
		output_str = output_str + single_prediciton + '\n';
	}

	document.getElementById('output').innerText = output_str;
	//cv.imshow('canvasOutput', img);
};

function onOpenCvReady() 
{
  console.log('OpenCV.js is ready.');
}

function onTensorFlowReady() 
{
  console.log('TensorFlow.js is ready.');
}
