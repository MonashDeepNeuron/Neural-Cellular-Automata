import startingPatterns from "../patterns/startingPatterns";
import { useDispatch, useSelector } from "react-redux";
import { useState } from "react";
import { changeTemplate, resetStep, toggleRunning } from "../store/webGPUSlice";

export default function TemplateDropdown() {
    const dispatch = useDispatch();
    const [selectedValue, setSelectedValue] = useState('');

    const running = useSelector((state) => state.webGPU.running);
    console.log('running from templateDropdown,', running);

    const handleChange = (event) => {
        setSelectedValue(prev => event.target.value);
    }
    const handleTemplateChange = (event) => {
        event.preventDefault();
        console.log('handleTemplateChange');
        dispatch(changeTemplate(selectedValue));
        dispatch(resetStep());
    }
    return (
        <div className="flex bg-pink-100 items-center p-8">

            <form onSubmit={handleTemplateChange} className="flex flex-col items-center justify-center gap-4">
                <label htmlFor="dropdown" className="text-lg font-semibold">
                    Choose Template:
                </label>
                <select
                    id="dropdown"
                    value={selectedValue}
                    onChange={handleChange}
                    className="p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring focus:ring-blue-300"
                >
                    <option value="" disabled>
                        Select a Template
                    </option>
                    {startingPatterns.map((pattern, index) => (
                        <option key={pattern.name} value={index}>
                            {pattern.name}
                        </option>
                    ))}
                </select>
                <button
                    type="submit"
                    onClick={handleTemplateChange}
                    className="bg-pink-500 text-white px-4 py-2 rounded-md shadow hover:bg-pink-600 focus:ring focus:ring-pink-300"
                >
                    Submit
                </button>

            </form>
        </div>)
    // return <form onSubmit={handleTemplateChange}>
    //     <label htmlFor="dropdown">Choose an option:</label>
    //     <select id="dropdown" value={selectedValue} onChange={handleChange}>
    //         <option value="" disabled>
    //             -- Select a Template --
    //         </option>
    //         {startingPatterns.map((pattern, index) => (
    //             <option key={pattern.name} value={index}>
    //                 {pattern.name}
    //             </option>
    //         ))}
    //     </select>
    //     <button type="submit" onClick={handleTemplateChange}>Submit</button>
    // </form>
}