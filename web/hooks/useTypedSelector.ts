import { type TypedUseSelectorHook, useSelector } from 'react-redux';
import type { RootState } from '../store/store';

const useTypedSelector: TypedUseSelectorHook<RootState> = useSelector;
export default useTypedSelector;
