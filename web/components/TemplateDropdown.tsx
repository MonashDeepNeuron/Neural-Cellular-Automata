'use client';

import useTypedSelector from '@/hooks/useTypedSelector';
import { type ChangeEventHandler, type MouseEventHandler, useState } from 'react';
import { useDispatch } from 'react-redux';
import startingPatterns from '../patterns/startingPatterns';
import { changeTemplate, resetStep } from '../store/webGPUSlice';

export default function TemplateDropdown() {
	const dispatch = useDispatch();
	const template = useTypedSelector(state => state.webGPU.template);
	const [selectedValue, setSelectedValue] = useState(template);

	const handleChange: ChangeEventHandler<HTMLSelectElement> = event => {
		setSelectedValue(Number(event.target.value));
	};
	const handleTemplateChange: MouseEventHandler<HTMLButtonElement> = event => {
		event.preventDefault();
		dispatch(changeTemplate(selectedValue));
		dispatch(resetStep());
	};
	return (
		<div className='flex rounded-md shadow-lg bg-pink-100 items-center p-8'>
			<form onSubmit={e => e.preventDefault()} className='flex flex-col items-center justify-center gap-4'>
				<label htmlFor='dropdown' className='text-lg font-semibold'>
					Starting Pattern
				</label>
				<select
					id='dropdown'
					value={selectedValue}
					onChange={handleChange}
					className='p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring focus:ring-blue-300'
				>
					<option value='' disabled>
						Select a Template
					</option>
					{startingPatterns.map((pattern, index) => (
						<option key={pattern.name} value={index}>
							{pattern.name}
						</option>
					))}
				</select>
				<button
					type='submit'
					onClick={handleTemplateChange}
					className='bg-pink-500 text-white px-4 py-2 rounded-md shadow hover:bg-pink-600 focus:ring focus:ring-pink-300'
				>
					Submit
				</button>
			</form>
		</div>
	);
}
