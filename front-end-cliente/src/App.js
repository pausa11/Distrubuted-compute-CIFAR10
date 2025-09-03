import './App.css';

function App() {
  return (
    <div id='main-container' className="w-screen h-screen">
      <h1 id='tittle-1' className='text-3xl font-bold underline'>
        sistema de computo distribuido
      </h1>

      <h2 id='tittle-2' className='text-2xl font-bold underline'>
        elige numero de batches para el dataset de CIFAR-10
      </h2>


      <form id='form' className='flex flex-col items-center'>
        <label id='label' className='text-xl font-bold underline mb-4'>
          numero de batches:
        </label>
        <input id='input' className='border border-gray-300 rounded-md p-2 mb-4' type="number" name="batches" min="128" max="1024" step="128" defaultValue="128" />
        <button id='submit-button' className='bg-blue-500 text-white rounded-md p-2 hover:bg-blue-600' type="submit">enviar</button>
      </form>


    </div>
  );
}

export default App;
